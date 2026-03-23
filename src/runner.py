"""
Quantitative comparison: SLAM with vs without loop closure.

For each dataset, runs SLAM twice (loop closure on/off) and reports:
  - Start-to-end pose distance
  - Final graph error
  - Number of loop closures detected
  - Trajectory length
"""

import numpy as np
import math
from icp_full import (
    SLAMSystem, JSONLoader, Keyframe, RobotPose,
    LOOP_CLOSURE
)
import icp_full  # so we can toggle the global


def compute_metrics(slam: SLAMSystem) -> dict:
    kfs = slam.all_keyframes
    start = kfs[0].pose
    end = kfs[-1].pose

    start_end_dist = math.sqrt((start.x - end.x)**2 + (start.y - end.y)**2)
    start_end_dtheta = abs(start.theta - end.theta)

    # trajectory length
    traj_len = 0.0
    for i in range(1, len(kfs)):
        dx = kfs[i].pose.x - kfs[i-1].pose.x
        dy = kfs[i].pose.y - kfs[i-1].pose.y
        traj_len += math.sqrt(dx*dx + dy*dy)

    # graph error (use result if optimised, otherwise initial)
    values = slam.result if slam.result is not None else slam.initial
    graph_error = slam.graph.error(values)

    return {
        "start_end_dist": start_end_dist,
        "start_end_dtheta_deg": math.degrees(start_end_dtheta),
        "graph_error": graph_error,
        "num_keyframes": len(kfs),
        "num_loop_closures": len(slam.loop_closure_constraints),
        "trajectory_length": traj_len,
    }


def run_slam_with_metrics(lidar_file: str, loop_closure: bool) -> dict:
    icp_full.LOOP_CLOSURE = loop_closure
    loader = JSONLoader(lidar_file)
    scans = loader.load_scans()
    slam = SLAMSystem()

    for i, scan in enumerate(scans):
        if (i + 1) % 500 == 0 or i == len(scans) - 1:
            print(f"  scan {i+1}/{len(scans)}", end="\r")
        slam.add_scan(scan)

    slam.finalise()
    print()

    return compute_metrics(slam)


if __name__ == "__main__":
    files = ["classroom_proper", "level7", "riverside"]

    results = {}
    for lidar_file in files:
        results[lidar_file] = {}
        for lc_enabled in [False, True]:
            label = "with LC" if lc_enabled else "no LC"
            print(f"\n{'='*60}")
            print(f"  {lidar_file} — {label}")
            print(f"{'='*60}")
            results[lidar_file][label] = run_slam_with_metrics(lidar_file, lc_enabled)

    # ── Print table ──
    print("\n\n")
    print(f"{'Dataset':<22} {'Mode':<10} {'Start→End (m)':<16} {'Δθ (°)':<10} {'Graph Error':<14} {'Traj Len (m)':<14} {'LCs':<5}")
    print("-" * 95)

    for lidar_file in files:
        for label in ["no LC", "with LC"]:
            m = results[lidar_file][label]
            print(
                f"{lidar_file:<22} {label:<10} "
                f"{m['start_end_dist']:<16.4f} "
                f"{m['start_end_dtheta_deg']:<10.2f} "
                f"{m['graph_error']:<14.4f} "
                f"{m['trajectory_length']:<14.2f} "
                f"{m['num_loop_closures']:<5}"
            )
        print()

    # ── Also produce a LaTeX table ──
    print("\n% LaTeX table")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{l l r r r r}")
    print(r"\toprule")
    print(r"Dataset & Mode & Start$\to$End (m) & $\Delta\theta$ ($^\circ$) & Graph Error & Loop Closures \\")
    print(r"\midrule")

    for lidar_file in files:
        for label in ["no LC", "with LC"]:
            m = results[lidar_file][label]
            name = lidar_file.replace("_", r"\_")
            print(
                f"{name} & {label} & "
                f"{m['start_end_dist']:.3f} & "
                f"{m['start_end_dtheta_deg']:.1f} & "
                f"{m['graph_error']:.3f} & "
                f"{m['num_loop_closures']} \\\\"
            )
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Quantitative comparison of SLAM with and without loop closure.}")
    print(r"\label{tab:loop_closure_metrics}")
    print(r"\end{table}")