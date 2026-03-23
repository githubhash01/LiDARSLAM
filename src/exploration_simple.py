"""
Loop closure detection: dual-signal exploration.

Compares descriptor similarity and pose proximity against the first
keyframe to show that loop closures correspond to simultaneous peaks
in both signals.
"""

import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from icp_slam import *


# ── Config ───────────────────────────────────────────────────────────────
DESCRIPTOR_WINDOW = 10   # keyframes either side for aggregated descriptor
PROXIMITY_THRESHOLD = 3.0


# ── Descriptor ───────────────────────────────────────────────────────────

def scan_descriptor(points: np.ndarray) -> np.ndarray:
    """2D bearing-range histogram descriptor (36 x 24 = 864-dim)."""
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


def gather_local_points(kf: Keyframe, keyframes: list[Keyframe]) -> np.ndarray:
    """Aggregate world points from nearby keyframes into kf's local frame."""
    idx = kf.id
    lo = max(0, idx - DESCRIPTOR_WINDOW)
    hi = min(len(keyframes), idx + DESCRIPTOR_WINDOW + 1)
    T_inv = np.linalg.inv(kf.pose.as_matrix())
    chunks = []
    for k in keyframes[lo:hi]:
        pts_h = np.hstack([k.world_points, np.ones((len(k.world_points), 1))])
        local = (T_inv @ pts_h.T).T[:, :2]
        chunks.append(local)
    return np.vstack(chunks)


def keyframe_similarity(kf1: Keyframe, kf2: Keyframe, keyframes: list[Keyframe]) -> float:
    """Cosine similarity between aggregated bearing-range descriptors."""
    d1 = scan_descriptor(gather_local_points(kf1, keyframes))
    d2 = scan_descriptor(gather_local_points(kf2, keyframes))
    return float(np.dot(d1, d2))


def pose_distance(kf1: Keyframe, kf2: Keyframe) -> float:
    return math.sqrt((kf1.pose.x - kf2.pose.x)**2 + (kf1.pose.y - kf2.pose.y)**2)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_keyframes = pickle.load(open("riverside_keyframes.pkl", "rb"))
    ref = all_keyframes[0]

    # plot ref.points in world frame for sanity check with matplotlib
    plt.figure(figsize=(6, 6))
    plt.scatter(ref.world_points[:, 0], ref.world_points[:, 1], s=
        1, color="tab:blue", alpha=0.5)
    plt.title("Reference keyframe points in world frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reference_points.png", dpi=200)
    plt.show()  
    
    n = len(all_keyframes)
    print(f"Loaded {n} keyframes. Computing signals against keyframe 0...\n")

    distances = np.array([pose_distance(kf, ref) for kf in all_keyframes])
    proximities = 1.0 / (1.0 + distances)

    similarities = np.zeros(n)
    for i, kf in enumerate(all_keyframes):
        if distances[i] < PROXIMITY_THRESHOLD:
            similarities[i] = keyframe_similarity(kf, ref, all_keyframes)
        if i % 50 == 0:
            print(f"  [{i}/{n}]")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(similarities, label="Descriptor similarity", color="tab:blue", alpha=0.85)
    ax.plot(proximities, label="Pose proximity", color="tab:orange", alpha=0.85)
    ax.set_xlabel("Keyframe index")
    ax.set_ylabel("Value")
    ax.set_title("Descriptor similarity and pose proximity vs. first keyframe")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dual_signal_exploration.png", dpi=200)
    plt.show()
    print("Saved dual_signal_exploration.png")