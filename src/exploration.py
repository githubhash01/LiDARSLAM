"""
Pseudo-live loop closure detection — general multi-reference.

No assumption about returning to start. At each keyframe, check against
all sufficiently old keyframes that are spatially close. Use the proximity
gate to keep it cheap, then confirm with descriptor similarity.
"""

import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from icp_slam import *


# ── Config ───────────────────────────────────────────────────────────────
MIN_KEYFRAME_GAP = 30
PROXIMITY_THRESHOLD = 3.0
SIM_PEAK_WINDOW = 5
DESCRIPTOR_WINDOW = 10
MAX_CLOSURES = 4
MIN_CLOSURE_GAP = 30        # minimum index gap between consecutive closures
SIM_THRESHOLD = 0.15        # minimum descriptor similarity to consider


# ── Descriptor ───────────────────────────────────────────────────────────

def gather_local_points(
    kf: Keyframe, keyframes: list[Keyframe], window: int = DESCRIPTOR_WINDOW
) -> np.ndarray:
    idx = next(i for i, k in enumerate(keyframes) if k.id == kf.id)
    lo = max(0, idx - window)
    hi = min(len(keyframes), idx + window + 1)
    T_inv = np.linalg.inv(kf.pose.as_matrix())
    chunks = []
    for k in keyframes[lo:hi]:
        pts_h = np.hstack([k.world_points, np.ones((len(k.world_points), 1))])
        local = (T_inv @ pts_h.T).T[:, :2]
        chunks.append(local)
    return np.vstack(chunks)


def scan_descriptor(points: np.ndarray) -> np.ndarray:
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


def keyframe_similarity(kf1: Keyframe, kf2: Keyframe, keyframes: list[Keyframe]) -> float:
    pts1 = gather_local_points(kf1, keyframes)
    pts2 = gather_local_points(kf2, keyframes)
    return float(np.dot(scan_descriptor(pts1), scan_descriptor(pts2)))


def keyframe_pose_distance(kf1: Keyframe, kf2: Keyframe) -> float:
    return math.sqrt((kf1.pose.x - kf2.pose.x) ** 2 + (kf1.pose.y - kf2.pose.y) ** 2)


# ── Live detection ───────────────────────────────────────────────────────

def detect_loop_closures(keyframes: list[Keyframe]) -> list[tuple[int, int]]:
    """
    Scan keyframes sequentially. At each keyframe, find the best matching
    old keyframe (if any) using proximity + descriptor similarity.

    Returns list of (closure_idx, reference_idx) pairs.
    """
    n = len(keyframes)
    closures: list[tuple[int, int]] = []

    # for each keyframe, store its best (sim, ref_idx) against any old keyframe
    best_sim = np.zeros(n)
    best_ref = np.full(n, -1, dtype=int)

    print("Scanning keyframes...\n")

    for i in range(MIN_KEYFRAME_GAP, n):
        if len(closures) >= MAX_CLOSURES:
            break

        # find all old keyframes that are close enough spatially
        kf_i = keyframes[i]
        candidate_ref = -1
        candidate_sim = 0.0

        for j in range(0, i - MIN_KEYFRAME_GAP):
            d = keyframe_pose_distance(kf_i, keyframes[j])
            if d >= PROXIMITY_THRESHOLD:
                continue

            s = keyframe_similarity(kf_i, keyframes[j], keyframes)
            if s > candidate_sim:
                candidate_sim = s
                candidate_ref = j

        best_sim[i] = candidate_sim
        best_ref[i] = candidate_ref

        if i % 50 == 0:
            print(f"  [{i}/{n}] best_sim={candidate_sim:.3f} ref_idx={candidate_ref}")

    # now detect peaks in best_sim
    print("\nDetecting peaks...\n")

    last_closure_idx = -MIN_CLOSURE_GAP * 2

    for c in range(MIN_KEYFRAME_GAP, n):
        if len(closures) >= MAX_CLOSURES:
            break
        if c - last_closure_idx < MIN_CLOSURE_GAP:
            continue
        if best_sim[c] < SIM_THRESHOLD:
            continue
        if best_ref[c] < 0:
            continue

        lo = max(0, c - SIM_PEAK_WINDOW)
        hi = min(n, c + SIM_PEAK_WINDOW + 1)

        # must be near the local peak
        local_max = np.max(best_sim[lo:hi])
        if best_sim[c] < 0.95 * local_max:
            continue

        ref_idx = best_ref[c]
        closures.append((c, ref_idx))
        last_closure_idx = c
        print(
            f"  ✓ Closure {len(closures)}: idx={c} (KF {keyframes[c].id}) "
            f"↔ ref idx={ref_idx} (KF {keyframes[ref_idx].id}), "
            f"sim={best_sim[c]:.3f}"
        )

    return closures


# ── Visualisation ────────────────────────────────────────────────────────

def plot_results(keyframes: list[Keyframe], closures: list[tuple[int, int]]):
    n = len(keyframes)
    colours = ["red", "green", "purple", "darkorange"]

    # ── Plot 1: best similarity signal with detected peaks ──
    # recompute best_sim for plotting
    best_sim = np.zeros(n)
    best_ref = np.full(n, -1, dtype=int)
    for i in range(MIN_KEYFRAME_GAP, n):
        kf_i = keyframes[i]
        for j in range(0, i - MIN_KEYFRAME_GAP):
            d = keyframe_pose_distance(kf_i, keyframes[j])
            if d >= PROXIMITY_THRESHOLD:
                continue
            s = keyframe_similarity(kf_i, keyframes[j], keyframes)
            if s > best_sim[i]:
                best_sim[i] = s
                best_ref[i] = j

    # min distance to any old keyframe (beyond gap)
    min_dist = np.full(n, np.inf)
    for i in range(MIN_KEYFRAME_GAP, n):
        for j in range(0, i - MIN_KEYFRAME_GAP):
            d = keyframe_pose_distance(keyframes[i], keyframes[j])
            if d < min_dist[i]:
                min_dist[i] = d
    proximities = 1.0 / (1.0 + min_dist)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(best_sim, label="Best Descriptor Similarity", alpha=0.8)
    ax.plot(proximities, label="Best Pose Proximity", alpha=0.8)

    for idx, (ci, ri) in enumerate(closures):
        colour = colours[idx % len(colours)]
        ax.axvline(ci, color=colour, linestyle="--", alpha=0.7,
                   label=f"Closure {idx+1}" if idx == 0 else f"Closure {idx+1}")
        ax.annotate(
            f"KF {keyframes[ci].id}↔{keyframes[ri].id}",
            xy=(ci, best_sim[ci]),
            xytext=(ci + 8, min(best_sim[ci] + 0.1, 0.95)),
            arrowprops=dict(arrowstyle="->", color=colour),
            color=colour, fontsize=8,
        )

    ax.set_xlabel("Keyframe Index")
    ax.set_ylabel("Value")
    ax.set_title("General Loop Closure Detection (any reference)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # ── Plot 2: trajectory with closure links ──
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    xs = [kf.pose.x for kf in keyframes]
    ys = [kf.pose.y for kf in keyframes]
    ax2.plot(xs, ys, "b-", alpha=0.3, linewidth=0.5)
    ax2.scatter(xs, ys, c=range(n), cmap="viridis", s=3, zorder=2)

    for idx, (ci, ri) in enumerate(closures):
        colour = colours[idx % len(colours)]
        ax2.plot(
            [keyframes[ri].pose.x, keyframes[ci].pose.x],
            [keyframes[ri].pose.y, keyframes[ci].pose.y],
            color=colour, linewidth=2, linestyle="--", zorder=3,
        )
        ax2.scatter(
            [keyframes[ci].pose.x], [keyframes[ci].pose.y],
            color=colour, s=80, zorder=4, marker="x",
            label=f"Closure {idx+1}: KF{keyframes[ci].id}↔KF{keyframes[ri].id}",
        )
        ax2.scatter(
            [keyframes[ri].pose.x], [keyframes[ri].pose.y],
            color=colour, s=80, zorder=4, marker="o",
        )

    ax2.scatter([xs[0]], [ys[0]], color="black", s=120, zorder=5, marker="*", label="Start")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Trajectory with Detected Loop Closures")
    ax2.legend()
    ax2.axis("equal")
    plt.tight_layout()
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_keyframes = pickle.load(open("level7_keyframes.pkl", "rb"))
    print(f"Loaded {len(all_keyframes)} keyframes\n")

    closures = detect_loop_closures(all_keyframes)

    print(f"\n{'='*50}")
    print(f"Detected {len(closures)} loop closure(s):")
    for i, (ci, ri) in enumerate(closures):
        print(f"  {i+1}. KF {all_keyframes[ci].id} (idx={ci}) ↔ KF {all_keyframes[ri].id} (idx={ri})")

    if closures:
        plot_results(all_keyframes, closures)

