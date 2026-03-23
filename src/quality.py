"""
Beam quality exploration across datasets.

Loads pickled keyframes and plots the distribution of beam quality
values to understand how quality differs between good and bad datasets.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from icp_full import * 


DATASETS = {
    # "Classroom Proper": "classroom_proper_keyframes.pkl",
    # "Level 7 Foyer": "level7_keyframes.pkl",
    "Riverside Cafe": "riverside_keyframes.pkl",
    "Bartlett": "bartlett_keyframes.pkl",
    "Car Park": "outdoor_keyframes.pkl",
}


def get_beam_qualities(keyframes) -> list[int]:
    qualities = []
    for kf in keyframes:
        for beam in kf.pointcloud_lidar.scan.beams:
            qualities.append(beam.quality)
    return qualities


def get_quality_vs_range(keyframes) -> tuple[list[int], list[float]]:
    qualities = []
    ranges = []
    for kf in keyframes:
        for beam in kf.pointcloud_lidar.scan.beams:
            qualities.append(beam.quality)
            ranges.append(beam.range_mm)
    return qualities, ranges


if __name__ == "__main__":
    all_data = {}
    for name, path in DATASETS.items():
        try:
            kfs = pickle.load(open(path, "rb"))
            all_data[name] = kfs
            print(f"Loaded {name}: {len(kfs)} keyframes")
        except FileNotFoundError:
            print(f"Skipping {name}: {path} not found")

    n = len(all_data)
    if n == 0:
        print("No datasets found.")
        exit()

    # ── Plot 1: quality histograms side by side ──
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i, (name, kfs) in enumerate(all_data.items()):
        ax = axes[0, i]
        quals = get_beam_qualities(kfs)
        ax.hist(quals, bins=range(0, max(quals) + 2), edgecolor="black", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Beam quality")
        ax.set_ylabel("Count")
        median_q = np.median(quals)
        mean_q = np.mean(quals)
        ax.axvline(median_q, color="red", linestyle="--", label=f"median={median_q:.0f}")
        ax.axvline(mean_q, color="orange", linestyle="--", label=f"mean={mean_q:.1f}")
        ax.legend(fontsize=8)
    plt.suptitle("Beam Quality Distribution by Dataset")
    plt.tight_layout()
    plt.savefig("beam_quality_histograms.png", dpi=200)
    plt.show()

    # ── Plot 2: quality vs range scatter ──
    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i, (name, kfs) in enumerate(all_data.items()):
        ax = axes2[0, i]
        quals, ranges = get_quality_vs_range(kfs)
        # subsample for plotting speed
        idx = np.random.choice(len(quals), min(50000, len(quals)), replace=False)
        ax.scatter(np.array(ranges)[idx], np.array(quals)[idx], s=1, alpha=0.3)
        ax.set_title(name)
        ax.set_xlabel("Range (mm)")
        ax.set_ylabel("Quality")
    plt.suptitle("Beam Quality vs Range")
    plt.tight_layout()
    plt.savefig("beam_quality_vs_range.png", dpi=200)
    plt.show()

    # ── Print summary stats ──
    print(f"\n{'Dataset':<22} {'Beams':<12} {'Mean Q':<10} {'Median Q':<10} {'% Q<10':<10} {'% Q<5':<10}")
    print("-" * 74)
    for name, kfs in all_data.items():
        quals = np.array(get_beam_qualities(kfs))
        total = len(quals)
        print(
            f"{name:<22} {total:<12} "
            f"{np.mean(quals):<10.1f} {np.median(quals):<10.0f} "
            f"{100 * np.sum(quals < 10) / total:<10.1f} "
            f"{100 * np.sum(quals < 5) / total:<10.1f}"
        )