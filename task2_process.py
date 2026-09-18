#!/usr/bin/env python3

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except ImportError:
    raise SystemExit(
        "Open3D is not installed. Run: pip3 install open3d"
    )


BASE = os.path.expanduser("~/corridor_ws/task2_data")
LOGS = os.path.expanduser("~/corridor_ws/task2_logs")

PCL_DIR = os.path.join(BASE, "B_noisy_add")

ODOM_FILE = os.path.join(LOGS, "odom.csv")
NOISY_ODOM_FILE = os.path.join(LOGS, "odom_noisy.csv")
EKF_FILE = os.path.join(LOGS, "ekf.csv")

ICP_FILE = os.path.join(LOGS, "icp_noisy.csv")
FINAL_EKF_FILE = os.path.join(LOGS, "ekf_with_lidar.csv")
PLOT_FILE = os.path.join(LOGS, "trajectory_comparison.png")


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def yaw_from_rotation(R):

    return np.arctan2(R[1, 0], R[0, 0])


def make_cloud(points):

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64)
    )

    return cloud


# ---------------------------------------------------------
# ICP
# ---------------------------------------------------------

def run_icp():

    files = sorted(
        glob.glob(os.path.join(PCL_DIR, "scan_*.npy"))
    )

    print("PCL directory:", PCL_DIR)
    print("Number of scans:", len(files))

    if len(files) < 2:
        raise RuntimeError("Not enough point clouds for ICP.")

    results = []

    accumulated = np.eye(4)

    trajectory = []

    trajectory.append(
        [0.0, 0.0, 0.0]
    )

    previous = np.load(files[0])

    # Remove invalid values
    previous = previous[
        np.isfinite(previous).all(axis=1)
    ]

    # Downsample using voxel grid
    previous_cloud = make_cloud(previous)

    previous_cloud = previous_cloud.voxel_down_sample(
        voxel_size=0.08
    )

    for i in range(1, len(files)):

        current = np.load(files[i])

        current = current[
            np.isfinite(current).all(axis=1)
        ]

        current_cloud = make_cloud(current)

        current_cloud = current_cloud.voxel_down_sample(
            voxel_size=0.08
        )

        # Previous scan -> current scan
        result = o3d.pipelines.registration.registration_icp(
            previous_cloud,
            current_cloud,
            1.0,
            np.eye(4),
            o3d.pipelines.registration.
            TransformationEstimationPointToPoint()
        )

        T = result.transformation

        dx = T[0, 3]
        dy = T[1, 3]
        dz = T[2, 3]

        dyaw = yaw_from_rotation(T[:3, :3])

        accumulated = accumulated @ T

        x = accumulated[0, 3]
        y = accumulated[1, 3]

        trajectory.append(
            [x, y, yaw_from_rotation(accumulated[:3, :3])]
        )

        results.append([
            i,
            dx,
            dy,
            dz,
            dyaw,
            result.fitness,
            result.inlier_rmse
        ])

        previous_cloud = current_cloud

        if i % 100 == 0:

            print(
                f"ICP {i}/{len(files)-1} "
                f"fitness={result.fitness:.3f} "
                f"RMSE={result.inlier_rmse:.4f}"
            )

    df = pd.DataFrame(
        results,
        columns=[
            "scan_id",
            "dx",
            "dy",
            "dz",
            "dyaw",
            "fitness",
            "rmse"
        ]
    )

    df.to_csv(ICP_FILE, index=False)

    trajectory = np.asarray(trajectory)

    return df, trajectory


# ---------------------------------------------------------
# Interpolate ICP to EKF timeline
# ---------------------------------------------------------

def integrate_icp_with_ekf(icp_traj, ekf):

    # Existing EKF is our odom + noisy IMU estimate.
    #
    # ICP trajectory starts at zero, therefore normalize
    # EKF trajectory to zero as well.

    ex = ekf["x"].to_numpy()
    ey = ekf["y"].to_numpy()
    eyaw = ekf["yaw"].to_numpy()

    ex = ex - ex[0]
    ey = ey - ey[0]

    # ICP trajectory is indexed by scan number.
    # Map scan index onto EKF sample index.

    icp_x = icp_traj[:, 0]
    icp_y = icp_traj[:, 1]
    icp_yaw = icp_traj[:, 2]

    ekf_index = np.linspace(
        0,
        len(ekf) - 1,
        len(icp_traj)
    )

    target_index = np.arange(len(ekf))

    lidar_x = np.interp(
        target_index,
        ekf_index,
        icp_x
    )

    lidar_y = np.interp(
        target_index,
        ekf_index,
        icp_y
    )

    lidar_yaw = np.interp(
        target_index,
        ekf_index,
        icp_yaw
    )

    # Lightweight fusion:
    # use EKF trajectory as prediction and ICP as
    # geometric correction.
    #
    # Weight can later be replaced with a proper
    # covariance-based Kalman update.

    alpha = 0.65

    fused_x = (
        alpha * ex +
        (1.0 - alpha) * lidar_x
    )

    fused_y = (
        alpha * ey +
        (1.0 - alpha) * lidar_y
    )

    fused_yaw = (
        alpha * eyaw +
        (1.0 - alpha) * lidar_yaw
    )

    result = pd.DataFrame({
        "time": ekf["time"],
        "x": fused_x,
        "y": fused_y,
        "yaw": fused_yaw
    })

    result.to_csv(
        FINAL_EKF_FILE,
        index=False
    )

    return result


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

def plot_all(odom, noisy, ekf, fused):

    plt.figure(figsize=(11, 8))

    plt.plot(
        odom["x"] - odom["x"].iloc[0],
        odom["y"] - odom["y"].iloc[0],
        label="Odometry",
        linewidth=2
    )

    plt.plot(
        noisy["x"] - noisy["x"].iloc[0],
        noisy["y"] - noisy["y"].iloc[0],
        label="Noisy odometry",
        linewidth=2
    )

    plt.plot(
        ekf["x"],
        ekf["y"],
        label="EKF (odom + IMU)",
        linewidth=2
    )

    plt.plot(
        fused["x"],
        fused["y"],
        label="EKF (odom + IMU + point cloud)",
        linewidth=2
    )

    plt.axvspan(
        10,
        15,
        alpha=0.12,
        label="Slip region"
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.title(
        "Trajectory comparison: noisy sensor fusion with LiDAR"
    )

    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()

    plt.savefig(
        PLOT_FILE,
        dpi=200
    )

    print()
    print("========================================")
    print("FINAL OUTPUT")
    print("========================================")
    print("ICP:", ICP_FILE)
    print("EKF + LiDAR:", FINAL_EKF_FILE)
    print("Plot:", PLOT_FILE)

    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading CSV data...")

    odom = pd.read_csv(ODOM_FILE)
    noisy = pd.read_csv(NOISY_ODOM_FILE)
    ekf = pd.read_csv(EKF_FILE)

    print("odom samples:", len(odom))
    print("noisy odom samples:", len(noisy))
    print("EKF samples:", len(ekf))

    print()
    print("Running ICP on B_noisy_add...")

    icp_df, icp_traj = run_icp()

    print()
    print("ICP completed.")

    print("Creating LiDAR-assisted trajectory...")

    fused = integrate_icp_with_ekf(
        icp_traj,
        ekf
    )

    print("Creating final plot...")

    plot_all(
        odom,
        noisy,
        ekf,
        fused
    )


if __name__ == "__main__":
    main()
