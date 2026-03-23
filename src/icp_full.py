from dataclasses import dataclass, field
import pickle
import numpy as np
import json
import math
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import gtsam
from gtsam import symbol

"""
GLOBAL CONSTANTS
"""
DATA_DIR = Path("/home/hashim/Desktop/Navigation/COMP0249_25-26/Courseworks/Coursework_02_-_SLAM/dataset/lidar/")
BLIND_SPOT_MIN = 135.0
BLIND_SPOT_MAX = 225.0
MINIMUM_RANGE_MM = 100.0
MINIMUM_QUALITY = 10

"""
CONSTANTS WE CHANGE FOR EXPERIMENTATION
"""
MAXIMUM_RANGE_MM: float = 12000.0
ANGULAR_RESOLUTION: int = 0  # keep every n-th beam (1 = all beams)
SCAN_RATE: int =0             # keep every n-th scan (1 = all scans)
VOXEL_SIZE: float = 0.0       # voxel grid cell size in meters (0 = no downsampling)
LOOP_CLOSURE: bool = True


# ---------- Robot Pose ----------
@dataclass
class RobotPose:
    x: float
    y: float
    theta: float  # heading in radians

    def rotation_matrix(self) -> np.ndarray:
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([
            [c, -s],
            [s,  c]
        ], dtype=float)

    def as_matrix(self) -> np.ndarray:
        R = self.rotation_matrix()
        t = np.array([self.x, self.y], dtype=float).reshape(2, 1)
        return np.block([
            [R, t],
            [np.zeros((1, 2)), np.array([[1.0]])]
        ])

    @staticmethod
    def from_matrix(T: np.ndarray):
        theta = np.arctan2(T[1, 0], T[0, 0])
        return RobotPose(
            x=float(T[0, 2]),
            y=float(T[1, 2]),
            theta=float(theta)
        )

@dataclass
class RigidBodyTransformation: 
    rotation: float  # in radians
    translation: np.ndarray  # shape (2,)

    def rotation_matrix(self) -> np.ndarray:
        c = np.cos(self.rotation)
        s = np.sin(self.rotation)
        return np.array([
            [c, -s],
            [s,  c]
        ], dtype=float)
    
    def as_matrix(self) -> np.ndarray:
        R = self.rotation_matrix()
        t = self.translation.reshape(2, 1)
        return np.block([
            [R, t],
            [np.zeros((1, 2)), np.array([[1]])]
        ])

def relative_motion(old_pose: RobotPose, new_pose: RobotPose) -> RigidBodyTransformation:
    T_old = old_pose.as_matrix()
    T_new = new_pose.as_matrix()
    delta_T = np.linalg.inv(T_old) @ T_new
    dx = delta_T[0, 2]
    dy = delta_T[1, 2]
    dtheta = np.arctan2(delta_T[1, 0], delta_T[0, 0])
    return RigidBodyTransformation(
        rotation=float(dtheta),
        translation=np.array([dx, dy], dtype=float)
    )
    
# ---------- Raw lidar data ----------
@dataclass
class Beam:
    quality: int
    bearing_degree: float
    range_mm: float

    @property
    def bearing_rad(self) -> float:
        return np.deg2rad(self.bearing_degree)

    @property
    def range_m(self) -> float:
        return self.range_mm / 1000.0

    @property
    def point_lidar(self) -> np.ndarray:
        r = self.range_m
        beta = self.bearing_rad
        x = r * np.cos(beta)
        y = r * np.sin(beta)
        return np.array([x, y], dtype=float)



# ---------- Scan ----------
@dataclass
class Scan:
    beams: list[Beam]

    def filter_beams(self) -> list[Beam]:
        ranges = np.array([b.range_mm for b in self.beams])
        angles = np.array([b.bearing_degree for b in self.beams])
        angle_mask = (angles < BLIND_SPOT_MIN) | (angles > BLIND_SPOT_MAX)
        range_mask = (ranges >= MINIMUM_RANGE_MM) & (ranges <= MAXIMUM_RANGE_MM)
        quality_mask = np.array([b.quality >= MINIMUM_QUALITY for b in self.beams])
        mask = angle_mask & range_mask & quality_mask
        filtered = [b for b, m in zip(self.beams, mask) if m]

        # Angular resolution downsampling: keep every n-th beam
        if ANGULAR_RESOLUTION > 1:
            filtered = filtered[::ANGULAR_RESOLUTION]

        return filtered

# ---------- Point Clouds ----------
class PointCloud:

    def __init__(self, scan: Scan):
        self.scan = scan
        self.lidar_points: np.ndarray = self._compute_lidar_points()

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        if voxel_size <= 0 or len(points) == 0:
            return points
        keys = np.floor(points / voxel_size).astype(int)
        cells: dict[tuple[int, int], list[np.ndarray]] = {}
        for i in range(len(keys)):
            k = (int(keys[i, 0]), int(keys[i, 1]))
            if k not in cells:
                cells[k] = []
            cells[k].append(points[i])
        return np.array([np.mean(v, axis=0) for v in cells.values()], dtype=float)

    def _compute_lidar_points(self) -> np.ndarray:
        filtered_beams = self.scan.filter_beams()
        points = np.array([b.point_lidar for b in filtered_beams], dtype=float)
        if len(points) > 0 and VOXEL_SIZE > 0:
            points = self._voxel_downsample(points, VOXEL_SIZE)
        return points
    
    def transform_points(self, pose: RobotPose) -> np.ndarray:
        c = np.cos(pose.theta)
        s = np.sin(pose.theta)
        R = np.array([[c, -s],
                    [s,  c]], dtype=float)
        t = np.array([pose.x, pose.y], dtype=float)
        return self.lidar_points @ R.T + t

# ---- SLAM Data Structures ----
@dataclass
class Keyframe:
    id: int
    pose: RobotPose
    pointcloud_lidar: PointCloud
    world_points: np.ndarray
    world_normals: np.ndarray

@dataclass
class OdometryConstraint: 
    from_id: int
    to_id: int
    relative_transform: RigidBodyTransformation

@dataclass
class LoopClosureConstraint:
    from_id: int
    to_id: int
    relative_transform: RigidBodyTransformation


# ── Loop Closure Detector ────────────────────────────────────────────────

class LoopClosureDetector:
    """
    Pseudo-live loop closure detection using bearing-range descriptors.
    
    Maintains a running buffer of per-keyframe similarity scores. At each
    new keyframe, computes descriptor similarity against all spatially 
    close old keyframes. Detects closures when a local peak in similarity
    is confirmed (delayed by a small window).
    """

    MIN_KEYFRAME_GAP = 30
    PROXIMITY_THRESHOLD = 3.0
    SIM_PEAK_WINDOW = 10
    DESCRIPTOR_WINDOW = 10
    MAX_CLOSURES = 10
    MIN_CLOSURE_GAP = 30
    SIM_THRESHOLD = 0.15

    def __init__(self):
        self.best_sim: list[float] = []
        self.best_ref: list[int] = []
        self.num_closures = 0
        self.last_closure_idx = -self.MIN_CLOSURE_GAP * 2

    # ── Descriptor computation ───────────────────────────────────────

    @staticmethod
    def _scan_descriptor(points: np.ndarray) -> np.ndarray:
        ranges = np.linalg.norm(points, axis=1)
        bearings = np.arctan2(points[:, 1], points[:, 0])
        hist, _, _ = np.histogram2d(
            bearings, ranges,
            bins=[36, 24],
            range=[[-np.pi, np.pi], [0, 12]],
        )
        hist = hist.ravel().astype(float)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist

    def _gather_local_points(
        self, kf: Keyframe, keyframes: list[Keyframe]
    ) -> np.ndarray:
        idx = kf.id
        lo = max(0, idx - self.DESCRIPTOR_WINDOW)
        hi = min(len(keyframes), idx + self.DESCRIPTOR_WINDOW + 1)
        T_inv = np.linalg.inv(kf.pose.as_matrix())
        chunks = []
        for k in keyframes[lo:hi]:
            pts_h = np.hstack([k.world_points, np.ones((len(k.world_points), 1))])
            local = (T_inv @ pts_h.T).T[:, :2]
            chunks.append(local)
        return np.vstack(chunks)

    def _similarity(
        self, kf1: Keyframe, kf2: Keyframe, keyframes: list[Keyframe]
    ) -> float:
        pts1 = self._gather_local_points(kf1, keyframes)
        pts2 = self._gather_local_points(kf2, keyframes)
        return float(np.dot(self._scan_descriptor(pts1), self._scan_descriptor(pts2)))

    # ── Main entry point — called after each keyframe is added ───────

    def check_for_closure(
        self, keyframes: list[Keyframe]
    ) -> tuple[int, int] | None:
        """
        Called after a new keyframe has been appended to `keyframes`.
        
        Returns (closure_kf_id, reference_kf_id) if a loop closure is
        detected, or None.
        """
        if self.num_closures >= self.MAX_CLOSURES:
            return None

        n = len(keyframes)
        new_kf = keyframes[-1]
        i = n - 1

        # compute best similarity for the new keyframe against all old ones
        candidate_sim = 0.0
        candidate_ref = -1

        if i >= self.MIN_KEYFRAME_GAP:
            for j in range(0, i - self.MIN_KEYFRAME_GAP):
                d = math.sqrt(
                    (new_kf.pose.x - keyframes[j].pose.x) ** 2
                    + (new_kf.pose.y - keyframes[j].pose.y) ** 2
                )
                if d >= self.PROXIMITY_THRESHOLD:
                    continue
                s = self._similarity(new_kf, keyframes[j], keyframes)
                if s > candidate_sim:
                    candidate_sim = s
                    candidate_ref = j

        self.best_sim.append(candidate_sim)
        self.best_ref.append(candidate_ref)

        # check if the candidate SIM_PEAK_WINDOW steps ago is a confirmed peak
        c = i - self.SIM_PEAK_WINDOW
        if c < self.MIN_KEYFRAME_GAP:
            return None
        if c - self.last_closure_idx < self.MIN_CLOSURE_GAP:
            return None
        if self.best_sim[c] < self.SIM_THRESHOLD:
            return None
        if self.best_ref[c] < 0:
            return None

        lo = max(0, c - self.SIM_PEAK_WINDOW)
        hi = min(len(self.best_sim), c + self.SIM_PEAK_WINDOW + 1)

        local_max = max(self.best_sim[lo:hi])
        if self.best_sim[c] < 0.95 * local_max:
            return None

        # confirmed loop closure
        ref_idx = self.best_ref[c]
        self.num_closures += 1
        self.last_closure_idx = c

        # return (keyframes[c].id, keyframes[ref_idx].id)

        # return the similiarity score and the proximity of the loop closure pair for debugging
        sim_score = self.best_sim[c]
        kf_c = keyframes[c]
        kf_ref = keyframes[ref_idx]
        proximity = math.sqrt(
            (kf_c.pose.x - kf_ref.pose.x) ** 2
            + (kf_c.pose.y - kf_ref.pose.y) ** 2
        )
        print(
            f"  Detected loop closure candidate: KF {kf_c.id} ↔ KF {kf_ref.id}, "
            f"sim={sim_score:.3f}, proximity={proximity:.2f} m"
        )
        return (keyframes[c].id, keyframes[ref_idx].id)

    def flush_pending(self, keyframes: list[Keyframe]) -> tuple[int, int] | None:
        """
        After all scans are processed, check the tail that couldn't be
        confirmed due to the peak window delay.
        """
        if self.num_closures >= self.MAX_CLOSURES:
            return None

        n = len(self.best_sim)
        # check the last SIM_PEAK_WINDOW candidates that weren't checked
        for c in range(max(self.MIN_KEYFRAME_GAP, n - self.SIM_PEAK_WINDOW), n):
            if c - self.last_closure_idx < self.MIN_CLOSURE_GAP:
                continue
            if self.best_sim[c] < self.SIM_THRESHOLD:
                continue
            if self.best_ref[c] < 0:
                continue

            lo = max(0, c - self.SIM_PEAK_WINDOW)
            hi = n

            local_max = max(self.best_sim[lo:hi])
            if self.best_sim[c] < 0.95 * local_max:
                continue

            ref_idx = self.best_ref[c]
            self.num_closures += 1
            self.last_closure_idx = c
            return (keyframes[c].id, keyframes[ref_idx].id)

        return None


class SLAMSystem:
    """
    Unified SLAM frontend + backend.

    The factor graph grows incrementally as keyframes arrive:
      - First keyframe  -> prior factor + initial value
      - Subsequent kf   -> odometry BetweenFactor + initial value
      - Loop closure     -> loop BetweenFactor + batch optimisation + pose update
    """

    # ----- ICP parameters -----
    ICP_MAX_ITER = 60    
    CORRESPONDENCE_THRESH = 0.4
    KEYFRAME_DIST_THRESH = 0.2 
    KEYFRAME_ANGLE_THRESH = 0.2
    LOCAL_MAP_SIZE = 20    
    MIN_CORRESPONDENCES = 20

    # ----- GTSAM noise models -----
    PRIOR_SIGMAS = np.array([0.00, 0.00, np.deg2rad(0.0)])
    ODOM_SIGMAS = np.array([0.10, 0.10, np.deg2rad(5.0)])
    LOOP_SIGMAS = np.array([0.05, 0.05, np.deg2rad(1.0)])

    def __init__(self):
        # Frontend state
        self.buffer_keyframes: list[Keyframe] = []
        self.all_keyframes: list[Keyframe] = []
        self.all_pose_constraints: list[OdometryConstraint] = []
        self.loop_closure_constraints: list[LoopClosureConstraint] = []

        self.robot_poses: list[RobotPose] = [RobotPose(0.0, 0.0, 0.0)]
        self.last_keyframe_pose: RobotPose = RobotPose(0.0, 0.0, 0.0)

        # Backend state (accumulates incrementally)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result: gtsam.Values | None = None

        # Noise model objects (created once)
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self.PRIOR_SIGMAS)
        self._odom_noise = gtsam.noiseModel.Diagonal.Sigmas(self.ODOM_SIGMAS)
        self._loop_noise = gtsam.noiseModel.Diagonal.Sigmas(self.LOOP_SIGMAS)

        # Loop closure detector
        self.loop_detector = LoopClosureDetector()

        # Event log for animation replay
        # Each entry: ("keyframe", kf_id, pose_at_insertion)
        #          or ("closure", closure_id, ref_id, {kf_id: RobotPose} snapshot after opt)
        self.event_log: list[tuple] = []

    # ------------------------------------------------------------------ #
    #                        helpers                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_pose2(pose: RobotPose) -> gtsam.Pose2:
        return gtsam.Pose2(pose.x, pose.y, pose.theta)

    @staticmethod
    def _to_relative_pose2(tf: RigidBodyTransformation) -> gtsam.Pose2:
        return gtsam.Pose2(float(tf.translation[0]),
                           float(tf.translation[1]),
                           float(tf.rotation))

    def _get_local_map_points_and_normals(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.buffer_keyframes:
            return (
                np.empty((0, 2), dtype=float),
                np.empty((0, 2), dtype=float)
            )
        active_points = np.vstack([kf.world_points for kf in self.buffer_keyframes])
        active_normals = np.vstack([kf.world_normals for kf in self.buffer_keyframes])
        return active_points, active_normals
    
    def _compute_normals(self, points: np.ndarray, k: int = 5) -> np.ndarray:
        if len(points) < k + 1:
            return np.zeros_like(points)

        neigh = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree")
        neigh.fit(points)
        _, indices_all = neigh.kneighbors(points)

        normals = np.zeros_like(points)

        for i in range(points.shape[0]):
            neighbor_points = points[indices_all[i]]
            centered = neighbor_points - np.mean(neighbor_points, axis=0)
            cov = np.dot(centered.T, centered) / k
            _, eig_vecs = np.linalg.eigh(cov)
            normal = eig_vecs[:, 0]
            if np.dot(normal, points[i]) < 0:
                normal = -normal
            normals[i] = normal

        return normals

    # ------------------------------------------------------------------ #
    #                        ICP                                          #
    # ------------------------------------------------------------------ #

    def _solve_point_to_plane(self, src_points: np.ndarray, dst_points: np.ndarray, dst_normals: np.ndarray) -> RigidBodyTransformation:
        A = []
        b = []

        for i in range(src_points.shape[0]):
            s = src_points[i]
            d = dst_points[i]
            n = dst_normals[i]
            cross_term = s[0] * n[1] - s[1] * n[0]
            A.append([cross_term, n[0], n[1]])
            b.append(np.dot(d - s, n))

        if len(A) == 0:
            return RigidBodyTransformation(0.0, np.zeros(2, dtype=float))

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return RigidBodyTransformation(
            rotation=float(x[0]),
            translation=np.array([x[1], x[2]], dtype=float)
        )

    def _icp_scan_to_map(self, current_pc: PointCloud, init_pose: RobotPose) -> RobotPose:
        active_points, active_normals = self._get_local_map_points_and_normals()

        if current_pc.lidar_points.shape[0] == 0 or active_points.shape[0] == 0:
            return init_pose

        current_pose_matrix = init_pose.as_matrix()

        neigh = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        neigh.fit(active_points)

        src_local = current_pc.lidar_points
        src_h = np.ones((3, src_local.shape[0]), dtype=float)
        src_h[:2, :] = src_local.T

        for _ in range(self.ICP_MAX_ITER):
            src_world_h = current_pose_matrix @ src_h
            src_world = src_world_h[:2, :].T

            distances, indices = neigh.kneighbors(src_world, return_distance=True)
            distances = distances.ravel()
            indices = indices.ravel()

            mask = distances < self.CORRESPONDENCE_THRESH
            if np.sum(mask) < self.MIN_CORRESPONDENCES:
                break

            src_valid = src_world[mask]
            dst_valid = active_points[indices[mask]]
            normals_valid = active_normals[indices[mask]]

            delta = self._solve_point_to_plane(src_valid, dst_valid, normals_valid)
            current_pose_matrix = delta.as_matrix() @ current_pose_matrix

        return RobotPose.from_matrix(current_pose_matrix)
    
    # ------------------------------------------------------------------ #
    #         Graph management                                            #
    # ------------------------------------------------------------------ #

    def _add_node_to_graph(self, kf: Keyframe) -> None:
        key = symbol('x', kf.id)
        self.initial.insert(key, self._to_pose2(kf.pose))

    def _add_prior_factor(self, kf: Keyframe) -> None:
        self.graph.add(
            gtsam.PriorFactorPose2(
                symbol('x', kf.id),
                self._to_pose2(kf.pose),
                self._prior_noise
            )
        )

    def _add_odometry_factor(self, constraint: OdometryConstraint) -> None:
        self.graph.add(
            gtsam.BetweenFactorPose2(
                symbol('x', constraint.from_id),
                symbol('x', constraint.to_id),
                self._to_relative_pose2(constraint.relative_transform),
                self._odom_noise
            )
        )

    def _add_loop_closure_factor(self, constraint: LoopClosureConstraint) -> None:
        self.graph.add(
            gtsam.BetweenFactorPose2(
                symbol('x', constraint.from_id),
                symbol('x', constraint.to_id),
                self._to_relative_pose2(constraint.relative_transform),
                self._loop_noise
            )
        )

    def _optimise_graph(self) -> None:
        """
        Run batch LM optimisation, then push corrected poses back into
        every keyframe and refresh cached world points/normals.
        """
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial, params
        )
        print(f"  graph error before opt: {self.graph.error(self.initial):.6f}")
        self.result = optimizer.optimize()
        print(f"  graph error after  opt: {self.graph.error(self.result):.6f}")

        # Push corrected poses back into ALL keyframes
        for kf in self.all_keyframes:
            p = self.result.atPose2(symbol('x', kf.id))
            kf.pose = RobotPose(x=p.x(), y=p.y(), theta=p.theta())
            kf.world_points = kf.pointcloud_lidar.transform_points(kf.pose)
            kf.world_normals = self._compute_normals(kf.world_points)

        # Update running state so next ICP init is consistent
        self.last_keyframe_pose = self.all_keyframes[-1].pose
        self.robot_poses[-1] = self.all_keyframes[-1].pose

        # Keep initial values consistent with optimised result
        self.initial = self.result

    # ------------------------------------------------------------------ #
    #                    Loop closure integration                         #
    # ------------------------------------------------------------------ #

    def _try_loop_closure(self, closure_id: int, reference_id: int) -> None:
        """
        Given a detected loop closure pair, add a constraint asserting
        the two keyframes are at the same pose, then optimise.
        """
        lc = LoopClosureConstraint(
            from_id=closure_id,
            to_id=reference_id,
            relative_transform=RigidBodyTransformation(
                rotation=0.0, translation=np.zeros(2)
            ),
        )
        self.loop_closure_constraints.append(lc)
        self._add_loop_closure_factor(lc)

        print(f"\n  Added loop closure: KF {closure_id} ↔ KF {reference_id}")

        self._optimise_graph()

        # Log corrected poses for animation replay
        corrected = {kf.id: RobotPose(kf.pose.x, kf.pose.y, kf.pose.theta)
                     for kf in self.all_keyframes}
        self.event_log.append(("closure", closure_id, reference_id, corrected))

    # ------------------------------------------------------------------ #
    #                    Keyframe insertion                                #
    # ------------------------------------------------------------------ #

    def _add_keyframe(self, current_pc: PointCloud, pose: RobotPose) -> None:
        new_id = len(self.all_keyframes)

        world_points = current_pc.transform_points(pose)
        world_normals = self._compute_normals(world_points)
        new_keyframe = Keyframe(
            id=new_id,
            pose=pose,
            pointcloud_lidar=current_pc,
            world_points=world_points,
            world_normals=world_normals,
        )

        # --- Graph: add node ---
        self._add_node_to_graph(new_keyframe)

        if len(self.all_keyframes) == 0:
            self._add_prior_factor(new_keyframe)
        else:
            # Odometry factor
            prev_kf = self.all_keyframes[-1]
            rel = relative_motion(self.last_keyframe_pose, pose)
            odom_constraint = OdometryConstraint(
                from_id=prev_kf.id,
                to_id=new_keyframe.id,
                relative_transform=rel
            )
            self.all_pose_constraints.append(odom_constraint)
            self._add_odometry_factor(odom_constraint)

        # Bookkeeping
        self.all_keyframes.append(new_keyframe)
        self.buffer_keyframes.append(new_keyframe)
        self.last_keyframe_pose = new_keyframe.pose

        if len(self.buffer_keyframes) > self.LOCAL_MAP_SIZE:
            self.buffer_keyframes.pop(0)

        # Log for animation replay
        self.event_log.append(("keyframe", new_id, RobotPose(pose.x, pose.y, pose.theta)))

        # --- Live loop closure detection ---
        if LOOP_CLOSURE:
            result = self.loop_detector.check_for_closure(self.all_keyframes)
            if result is not None:
                closure_id, ref_id = result
                self._try_loop_closure(closure_id, ref_id)

    # ------------------------------------------------------------------ #
    #                        Public API                                   #
    # ------------------------------------------------------------------ #

    def add_scan(self, scan: Scan) -> None:
        current_pc = PointCloud(scan)
        if current_pc.lidar_points.shape[0] == 0:
            return

        current_pose_guess = self.robot_poses[-1]

        if len(self.buffer_keyframes) == 0:
            self._add_keyframe(current_pc, current_pose_guess)
            return

        refined_pose = self._icp_scan_to_map(current_pc, current_pose_guess)
        self.robot_poses.append(refined_pose)

        rel = relative_motion(self.last_keyframe_pose, refined_pose)
        dist_moved = np.linalg.norm(rel.translation)
        dtheta = rel.rotation

        if dist_moved > self.KEYFRAME_DIST_THRESH or abs(dtheta) > self.KEYFRAME_ANGLE_THRESH:
            self._add_keyframe(current_pc, refined_pose)

    def finalise(self) -> None:
        """Call after all scans to check for closures in the tail buffer."""
        if not LOOP_CLOSURE:
            return
        result = self.loop_detector.flush_pending(self.all_keyframes)
        if result is not None:
            closure_id, ref_id = result
            self._try_loop_closure(closure_id, ref_id)

    # ------------------------------------------------------------------ #
    #                    Result accessors                                 #
    # ------------------------------------------------------------------ #

    @property
    def optimized_trajectory(self) -> np.ndarray:
        return np.array([[kf.pose.x, kf.pose.y] for kf in self.all_keyframes], dtype=float)

    @property
    def optimized_map_points(self) -> np.ndarray:
        all_points = []
        for kf in self.all_keyframes:
            world_points = kf.pointcloud_lidar.transform_points(kf.pose)
            all_points.append(world_points)
        return np.vstack(all_points)


# ------------------------------------------------------------------ #
#                   Occupancy Grid                                     #
# ------------------------------------------------------------------ #

class OccupancyGrid:
    """
    Log-odds occupancy grid built from SLAM keyframes via Bresenham ray-casting.

    Cells along each ray are marked free (log-odds decremented), the endpoint
    cell is marked occupied (log-odds incremented). Stores log-odds internally;
    call as_probability() for a [0, 1] occupancy map.
    """

    def __init__(self, resolution: float = 0.05, padding: float = 2.0):
        self.resolution = resolution
        self.padding = padding

        self.l_free = -0.1
        self.l_occ = 0.5
        self.l_max = 8.0
        self.l_min = -8.0

        self.grid: np.ndarray | None = None
        self.origin_x: float = 0.0
        self.origin_y: float = 0.0
        self.width: int = 0
        self.height: int = 0

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        cx, cy = x0, y0
        while True:
            if cx == x1 and cy == y1:
                break
            cells.append((cx, cy))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
        return cells

    def allocate_grid(self, all_points: np.ndarray) -> None:
        """Pre-allocate grid to contain the given point cloud (union of all poses)."""
        x_min, y_min = all_points.min(axis=0) - self.padding
        x_max, y_max = all_points.max(axis=0) + self.padding
        self.origin_x = x_min
        self.origin_y = y_min
        self.width = int(np.ceil((x_max - x_min) / self.resolution))
        self.height = int(np.ceil((y_max - y_min) / self.resolution))
        self.grid = np.zeros((self.width, self.height), dtype=np.float32)

    def clear_grid(self) -> None:
        if self.grid is not None:
            self.grid[:] = 0.0

    def integrate_keyframe(self, kf: Keyframe, pose: RobotPose | None = None) -> None:
        if self.grid is None:
            raise RuntimeError("Call allocate_grid() before integrate_keyframe().")
        pose = pose if pose is not None else kf.pose
        robot_gx, robot_gy = self.world_to_grid(pose.x, pose.y)
        world_hits = kf.pointcloud_lidar.transform_points(pose)

        for hit in world_hits:
            hit_gx, hit_gy = self.world_to_grid(hit[0], hit[1])
            for cx, cy in self._bresenham(robot_gx, robot_gy, hit_gx, hit_gy):
                if self._in_bounds(cx, cy):
                    self.grid[cx, cy] = max(self.l_min, self.grid[cx, cy] + self.l_free)
            if self._in_bounds(hit_gx, hit_gy):
                self.grid[hit_gx, hit_gy] = min(self.l_max, self.grid[hit_gx, hit_gy] + self.l_occ)

    def build(self, keyframes: list[Keyframe]) -> None:
        all_pts = np.vstack([kf.pointcloud_lidar.transform_points(kf.pose) for kf in keyframes])
        self.allocate_grid(all_pts)
        for kf in keyframes:
            self.integrate_keyframe(kf)

    def as_probability(self) -> np.ndarray:
        return 1.0 - 1.0 / (1.0 + np.exp(self.grid))


class Visualiser: 
    
    def __init__(self, slam: SLAMSystem):
        self.slam = slam

    def visualise(self, title: str):
        map_points = self.slam.optimized_map_points
        trajectory = self.slam.optimized_trajectory
        start_pose = trajectory[0]
        end_pose = trajectory[-1]
        print(f"Start Pose: x={start_pose[0]:.2f} m, y={start_pose[1]:.2f} m")            
        print(f"End Pose: x={end_pose[0]:.2f} m, y={end_pose[1]:.2f} m")            
        
        plt.figure(figsize=(10, 10))
        plt.scatter(map_points[:, 0], map_points[:, 1], s=1, c='blue', label='Map Points')
        plt.plot(trajectory[:, 0], trajectory[:, 1], c='red', label='Robot Trajectory')

        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='orange', s=100, marker='X', label='End')

        # plot the loop closures on the map
        for lc in self.slam.loop_closure_constraints:
            from_kf = self.slam.all_keyframes[lc.from_id]
            to_kf = self.slam.all_keyframes[lc.to_id]
            plt.plot(
                [from_kf.pose.x, to_kf.pose.x],
                [from_kf.pose.y, to_kf.pose.y],
                c='purple', linestyle='--', label='Loop Closure' if lc == self.slam.loop_closure_constraints[0] else ""
            )
        plt.title(title)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.legend()
        plt.grid()
        if LOOP_CLOSURE: 
            plt.savefig(f"{title}_loop_closure.png", dpi=300)
        else:
            plt.savefig(f"{title}_no_loop_closure.png", dpi=300)

    def visualise_occupancy(self, title: str, resolution: float = 0.05):
        """Build and display a static occupancy grid from the final optimised keyframes."""
        occ = OccupancyGrid(resolution=resolution)
        occ.build(self.slam.all_keyframes)
        prob = occ.as_probability()
        trajectory = self.slam.optimized_trajectory

        fig, ax = plt.subplots(figsize=(12, 12))
        extent = [
            occ.origin_x,
            occ.origin_x + occ.width * occ.resolution,
            occ.origin_y,
            occ.origin_y + occ.height * occ.resolution,
        ]
        ax.imshow(prob.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, extent=extent)
        ax.plot(trajectory[:, 0], trajectory[:, 1], c='red', linewidth=1, label='Trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=60, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='orange', s=60, marker='X', label='End')
        ax.set_title(f"{title} — Occupancy Grid ({resolution*100:.0f} cm/cell)")
        ax.set_aspect('equal')
        ax.axis('off')
        suffix = "loop_closure" if LOOP_CLOSURE else "no_loop_closure"
        plt.savefig(f"{title}_occupancy_{suffix}.png", dpi=300, bbox_inches='tight')
        print(f"Occupancy grid saved: {title}_occupancy_{suffix}.png")

    def animate_occupancy_live(
        self,
        title: str,
        resolution: float = 0.05,
        kf_per_frame: int = 3,
        fps: int = 20,
        rebuild_frames: int = 30,
    ):
        """
        Animate the occupancy grid using the SLAM event log so that loop
        closures appear as real-time corrections mid-build.

        For each "keyframe" event the grid is extended incrementally.
        For each "closure" event the grid clears and rapidly rebuilds from
        scratch with the corrected poses, then continues building.
        """
        from matplotlib.animation import FFMpegWriter

        event_log = self.slam.event_log
        keyframes = self.slam.all_keyframes
        kf_by_id = {kf.id: kf for kf in keyframes}

        # We need a grid large enough for all poses that will ever appear.
        # Collect world points under both the per-event poses AND final poses.
        all_pts_chunks = []
        for kf in keyframes:
            all_pts_chunks.append(kf.pointcloud_lidar.transform_points(kf.pose))
        # Also include the pre-closure poses recorded in keyframe events
        for ev in event_log:
            if ev[0] == "keyframe":
                _, kf_id, pose = ev
                all_pts_chunks.append(kf_by_id[kf_id].pointcloud_lidar.transform_points(pose))
        all_pts = np.vstack(all_pts_chunks)

        occ = OccupancyGrid(resolution=resolution)
        occ.allocate_grid(all_pts)

        extent = [
            occ.origin_x,
            occ.origin_x + occ.width * resolution,
            occ.origin_y,
            occ.origin_y + occ.height * resolution,
        ]

        # --- matplotlib setup ---
        fig, ax = plt.subplots(figsize=(10, 10))
        prob_init = occ.as_probability()
        im = ax.imshow(
            prob_init.T, origin='lower', cmap='gray_r',
            vmin=0, vmax=1, extent=extent,
        )
        traj_line, = ax.plot([], [], c='red', linewidth=1.5)
        robot_dot, = ax.plot([], [], 'o', c='lime', markersize=6)
        lc_lines = []  # will grow as closures appear
        title_text = ax.set_title("")
        ax.set_aspect('equal')
        ax.axis('off')

        # --- build frame schedule from event log ---
        # Walk the event log and group keyframe events into batches of kf_per_frame.
        # Each closure event gets its own rebuild sequence.
        #
        # pose_at[kf_id] tracks the pose currently used for each keyframe.
        # integrated[kf_id] tracks whether kf has been ray-cast into the grid.

        pose_at: dict[int, RobotPose] = {}
        integrated: set[int] = set()
        kf_order: list[int] = []  # order keyframes were added
        closures_so_far: list[tuple[int, int]] = []

        suffix = "loop_closure" if LOOP_CLOSURE else "no_loop_closure"
        out_path = f"{title}_occupancy_{suffix}.mp4"
        writer = FFMpegWriter(fps=fps, metadata={'title': title})

        def _update_display(label: str):
            im.set_data(occ.as_probability().T)
            if kf_order:
                traj_x = [pose_at[kid].x for kid in kf_order]
                traj_y = [pose_at[kid].y for kid in kf_order]
                traj_line.set_data(traj_x, traj_y)
                robot_dot.set_data([traj_x[-1]], [traj_y[-1]])
            title_text.set_text(label)

        def _draw_closure_lines():
            # Remove old lines
            for ln in lc_lines:
                ln.remove()
            lc_lines.clear()
            for (cid, rid) in closures_so_far:
                ln, = ax.plot(
                    [pose_at[cid].x, pose_at[rid].x],
                    [pose_at[cid].y, pose_at[rid].y],
                    c='cyan', linestyle='--', linewidth=1.5,
                )
                lc_lines.append(ln)

        frame_count = 0
        kf_event_buffer = []  # accumulate keyframe events between closures

        print(f"Rendering occupancy animation to {out_path} ...")

        with writer.saving(fig, out_path, dpi=150):
            for ev in event_log:
                if ev[0] == "keyframe":
                    _, kf_id, pose = ev
                    pose_at[kf_id] = pose
                    kf_order.append(kf_id)
                    kf_event_buffer.append(kf_id)

                    # Flush a frame every kf_per_frame keyframes
                    if len(kf_event_buffer) >= kf_per_frame:
                        for kid in kf_event_buffer:
                            if kid not in integrated:
                                occ.integrate_keyframe(kf_by_id[kid], pose=pose_at[kid])
                                integrated.add(kid)
                        kf_event_buffer.clear()
                        n_done = len(kf_order)
                        n_total = len(keyframes)
                        _update_display(f"{title} — Building map... KF {n_done}")
                        writer.grab_frame()
                        frame_count += 1

                elif ev[0] == "closure":
                    _, closure_id, ref_id, corrected_poses = ev

                    # Flush any remaining buffered keyframes with OLD poses
                    for kid in kf_event_buffer:
                        if kid not in integrated:
                            occ.integrate_keyframe(kf_by_id[kid], pose=pose_at[kid])
                            integrated.add(kid)
                    kf_event_buffer.clear()

                    _update_display(f"{title} — Building map... KF {len(kf_order)}")
                    writer.grab_frame()
                    frame_count += 1

                    # Pause briefly: flash the closure
                    closures_so_far.append((closure_id, ref_id))
                    _draw_closure_lines()
                    _update_display(f"{title} — Loop closure: KF {closure_id} ↔ KF {ref_id}")
                    for _ in range(int(fps * 0.75)):  # ~0.75 second pause
                        writer.grab_frame()
                        frame_count += 1

                    # Apply corrected poses
                    for kid, corrected_pose in corrected_poses.items():
                        pose_at[kid] = corrected_pose

                    # Rapid rebuild: clear grid, re-integrate all keyframes so far
                    occ.clear_grid()
                    integrated.clear()
                    n_so_far = len(kf_order)
                    kf_per_rebuild = max(1, n_so_far // rebuild_frames)
                    rebuild_steps = list(range(0, n_so_far, kf_per_rebuild))
                    if rebuild_steps[-1] != n_so_far - 1:
                        rebuild_steps.append(n_so_far - 1)

                    rebuild_integrated = 0
                    for step_end in rebuild_steps:
                        for k_idx in range(rebuild_integrated, step_end + 1):
                            kid = kf_order[k_idx]
                            occ.integrate_keyframe(kf_by_id[kid], pose=pose_at[kid])
                            integrated.add(kid)
                        rebuild_integrated = step_end + 1
                        _draw_closure_lines()
                        _update_display(f"{title} — Correcting map... {rebuild_integrated}/{n_so_far}")
                        writer.grab_frame()
                        frame_count += 1

            # Flush any remaining keyframes at the end
            for kid in kf_event_buffer:
                if kid not in integrated:
                    occ.integrate_keyframe(kf_by_id[kid], pose=pose_at[kid])
                    integrated.add(kid)
            kf_event_buffer.clear()
            _draw_closure_lines()
            _update_display(f"{title} — Final optimised occupancy grid")
            writer.grab_frame()
            frame_count += 1

            # Hold final frame for 2 seconds
            for _ in range(fps * 2):
                writer.grab_frame()
                frame_count += 1

        plt.close(fig)
        print(f"Video saved: {out_path} ({frame_count} frames)")

    def animate_map_points_live(
        self,
        title: str,
        kf_per_frame: int = 3,
        fps: int = 20,
        rebuild_frames: int = 30,
        point_size: float = 0.3,
    ):
        """
        Animate the point cloud map using the SLAM event log so that loop
        closures appear as real-time corrections mid-build.

        Mirrors animate_occupancy_live but renders scatter points instead
        of an occupancy grid.
        """
        from matplotlib.animation import FFMpegWriter

        event_log = self.slam.event_log
        keyframes = self.slam.all_keyframes
        kf_by_id = {kf.id: kf for kf in keyframes}

        # Compute axis limits from all points (both pre- and post-closure poses)
        all_pts_chunks = []
        for kf in keyframes:
            all_pts_chunks.append(kf.pointcloud_lidar.transform_points(kf.pose))
        for ev in event_log:
            if ev[0] == "keyframe":
                _, kf_id, pose = ev
                all_pts_chunks.append(kf_by_id[kf_id].pointcloud_lidar.transform_points(pose))
        all_pts = np.vstack(all_pts_chunks)
        padding = 2.0
        x_min, y_min = all_pts.min(axis=0) - padding
        x_max, y_max = all_pts.max(axis=0) + padding

        # --- matplotlib setup ---
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter([], [], s=point_size, c='blue', marker='.')
        traj_line, = ax.plot([], [], c='red', linewidth=1.5)
        robot_dot, = ax.plot([], [], 'o', c='lime', markersize=6)
        lc_lines = []
        title_text = ax.set_title("")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')

        # --- state tracking ---
        pose_at: dict[int, RobotPose] = {}
        kf_order: list[int] = []
        closures_so_far: list[tuple[int, int]] = []

        suffix = "loop_closure" if LOOP_CLOSURE else "no_loop_closure"
        out_path = f"{title}_map_points_{suffix}.mp4"
        writer = FFMpegWriter(fps=fps, metadata={'title': title})

        def _recompute_all_points() -> np.ndarray:
            if not kf_order:
                return np.empty((0, 2))
            chunks = [kf_by_id[kid].pointcloud_lidar.transform_points(pose_at[kid])
                      for kid in kf_order]
            return np.vstack(chunks)

        def _update_display(label: str, pts: np.ndarray | None = None):
            if pts is None:
                pts = _recompute_all_points()
            scatter.set_offsets(pts if len(pts) > 0 else np.empty((0, 2)))
            if kf_order:
                traj_x = [pose_at[kid].x for kid in kf_order]
                traj_y = [pose_at[kid].y for kid in kf_order]
                traj_line.set_data(traj_x, traj_y)
                robot_dot.set_data([traj_x[-1]], [traj_y[-1]])
            title_text.set_text(label)

        def _draw_closure_lines():
            for ln in lc_lines:
                ln.remove()
            lc_lines.clear()
            for (cid, rid) in closures_so_far:
                ln, = ax.plot(
                    [pose_at[cid].x, pose_at[rid].x],
                    [pose_at[cid].y, pose_at[rid].y],
                    c='cyan', linestyle='--', linewidth=1.5,
                )
                lc_lines.append(ln)

        frame_count = 0
        kf_event_buffer = []
        # Keep a running concatenation of world points for incremental updates
        accumulated_pts: list[np.ndarray] = []

        print(f"Rendering map-points animation to {out_path} ...")

        with writer.saving(fig, out_path, dpi=150):
            for ev in event_log:
                if ev[0] == "keyframe":
                    _, kf_id, pose = ev
                    pose_at[kf_id] = pose
                    kf_order.append(kf_id)
                    kf_event_buffer.append(kf_id)

                    if len(kf_event_buffer) >= kf_per_frame:
                        for kid in kf_event_buffer:
                            accumulated_pts.append(
                                kf_by_id[kid].pointcloud_lidar.transform_points(pose_at[kid])
                            )
                        kf_event_buffer.clear()
                        pts = np.vstack(accumulated_pts)
                        n_done = len(kf_order)
                        n_total = len(keyframes)
                        _update_display(f"{title} — Building map... KF {n_done}", pts)
                        writer.grab_frame()
                        frame_count += 1

                elif ev[0] == "closure":
                    _, closure_id, ref_id, corrected_poses = ev

                    # Flush buffered keyframes with old poses
                    for kid in kf_event_buffer:
                        accumulated_pts.append(
                            kf_by_id[kid].pointcloud_lidar.transform_points(pose_at[kid])
                        )
                    kf_event_buffer.clear()
                    pts = np.vstack(accumulated_pts) if accumulated_pts else np.empty((0, 2))
                    _update_display(f"{title} — Building map... KF {len(kf_order)}", pts)
                    writer.grab_frame()
                    frame_count += 1

                    # Flash the closure
                    closures_so_far.append((closure_id, ref_id))
                    _draw_closure_lines()
                    _update_display(f"{title} — Loop closure: KF {closure_id} ↔ KF {ref_id}", pts)
                    for _ in range(int(fps * 0.75)):
                        writer.grab_frame()
                        frame_count += 1

                    # Apply corrected poses
                    for kid, corrected_pose in corrected_poses.items():
                        pose_at[kid] = corrected_pose

                    # Rapid rebuild of all points with corrected poses
                    n_so_far = len(kf_order)
                    kf_per_rebuild = max(1, n_so_far // rebuild_frames)
                    rebuild_steps = list(range(0, n_so_far, kf_per_rebuild))
                    if rebuild_steps[-1] != n_so_far - 1:
                        rebuild_steps.append(n_so_far - 1)

                    rebuild_chunks: list[np.ndarray] = []
                    rebuild_idx = 0
                    for step_end in rebuild_steps:
                        for k_idx in range(rebuild_idx, step_end + 1):
                            kid = kf_order[k_idx]
                            rebuild_chunks.append(
                                kf_by_id[kid].pointcloud_lidar.transform_points(pose_at[kid])
                            )
                        rebuild_idx = step_end + 1
                        pts = np.vstack(rebuild_chunks)
                        _draw_closure_lines()
                        _update_display(f"{title} — Correcting map... {len(rebuild_chunks)}", pts)
                        writer.grab_frame()
                        frame_count += 1

                    # Replace accumulated with corrected points
                    accumulated_pts = rebuild_chunks

            # Flush remaining
            for kid in kf_event_buffer:
                accumulated_pts.append(
                    kf_by_id[kid].pointcloud_lidar.transform_points(pose_at[kid])
                )
            kf_event_buffer.clear()
            pts = np.vstack(accumulated_pts) if accumulated_pts else np.empty((0, 2))
            _draw_closure_lines()
            _update_display(f"{title} — Final optimised map", pts)
            writer.grab_frame()
            frame_count += 1

            # Hold final frame
            for _ in range(fps * 2):
                writer.grab_frame()
                frame_count += 1

        plt.close(fig)
        print(f"Video saved: {out_path} ({frame_count} frames)")


class JSONLoader:

    def __init__(self, filepath: str):
        self.filepath = f"{DATA_DIR}/{filepath}_LiDAR.json"

    def load_scans(self) -> list[Scan]:
        scans: list[Scan] = []
        with open(self.filepath, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                raw_scan = json.loads(line)

                beams: list[Beam] = []
                for raw_beam in raw_scan:
                    quality = int(raw_beam[0])
                    bearing_degree = float(raw_beam[1])
                    range_mm = float(raw_beam[2])
                    beam = Beam(quality=quality, bearing_degree=bearing_degree, range_mm=range_mm)
                    beams.append(beam)

                current_scan = Scan(beams=beams)
                scans.append(current_scan)

        return scans

def run_slam(lidar_file: str):
    loader = JSONLoader(lidar_file)
    scans = loader.load_scans()

    if SCAN_RATE > 1:
        scans = scans[::SCAN_RATE]
        print(f"Scan rate downsampling: keeping every {SCAN_RATE}-th scan ({len(scans)} scans remaining)")

    slam = SLAMSystem()

    for i, scan in enumerate(scans):
        print(f"Adding scan {i+1}/{len(scans)} to SLAM map...", end="\r")
        slam.add_scan(scan)

    # flush any pending loop closures in the tail buffer
    slam.finalise()

    print(f"\nNum keyframes: {len(slam.all_keyframes)}")
    print(f"Num odometry constraints: {len(slam.all_pose_constraints)}")
    print(f"Num loop closures: {len(slam.loop_closure_constraints)}")

    for i, lc in enumerate(slam.loop_closure_constraints):
        tf = lc.relative_transform
        print(
            f"LC {i}: {lc.from_id} -> {lc.to_id}, "
            f"dx={tf.translation[0]:.3f}, dy={tf.translation[1]:.3f}, "
            f"dtheta={np.rad2deg(tf.rotation):.2f} deg"
        )

    # print("\nSLAM processing complete. Visualising results...")
    visualiser = Visualiser(slam)
    visualiser.visualise(title=lidar_file)
    visualiser.visualise_occupancy(title=lidar_file, resolution=0.05)
    visualiser.animate_occupancy_live(
        title=lidar_file,
        resolution=0.05,
        kf_per_frame=3,
        fps=20,
    )
    visualiser.animate_map_points_live(
        title=lidar_file,
        kf_per_frame=3,
        fps=20,
    )

    return slam.all_keyframes

def main():
    files = ["classroom", "classroom_proper","level7", "riverside"]
    for lidar_file in files:
        print(f"\n\nRunning SLAM for {lidar_file} dataset...")
        all_keyframes = run_slam(lidar_file)

        # save the keyframes to a pickle file for later use in the animation script
        output_path = f"{lidar_file}_keyframes.pkl"
        with open(output_path, "wb") as fh:
            pickle.dump(all_keyframes, fh)
        print(f"Keyframes saved to {output_path}")

if __name__ == "__main__":
    main()