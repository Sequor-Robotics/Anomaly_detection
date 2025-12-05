import numpy as np
import tqdm
from pathlib import Path
import argparse
import os

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    np.loadtxt(data_with_single_line) ... (N,) → (1,N)
    """
    
    if arr.ndim == 1:
        return arr[None, :]
    
    return arr

def Read_parsed_data(scenario_path: str | Path):

    """
    For each scenario, transform parsed data into a dict.

    scenario_path: 
      - case 1) 시나리오 원본 폴더 (예: /data/scene01)
              -> 내부의 /data/scene01/scene01_parsed 를 자동 인식
      - case 2) 이미 *_parsed 폴더 자체 (예: /data/scene01/scene01_parsed)
              -> 그대로 사용
    """

    scenario_path = Path(scenario_path).resolve()

    # 1) parsed dir.
    candidate = scenario_path / f"{scenario_path.name}_parsed"
    if candidate.is_dir():
        parsed_dir = candidate
    else:
        parsed_dir = scenario_path

    print(f"[INFO] Using parsed dir: {parsed_dir}")

    lidar_dir = parsed_dir / "livox_lidar"
    scan_dir = parsed_dir / "scan_2d"
    imu_dir = parsed_dir / "imu"
    odom_dir = parsed_dir / "odom"
    pose_dir = parsed_dir / "pose"
    tf_dir = parsed_dir / "tf"

    # 1) 3D LiDAR frames (livox_lidar/*.npz)
    lidar_frames = []
    if lidar_dir.is_dir():
        for f in sorted(lidar_dir.glob("*.npz")):
            data = np.load(f)
            xyz = data["xyz"]              # (N,3)
            ts = float(data["timestamp"])  # scalar
            if "intensity" in data.files:
                intensity = data["intensity"]  # (N,) or (N,1)
            else:
                intensity = None

            lidar_frames.append(
                {
                    "path": f,
                    "timestamp": ts,
                    "xyz": xyz,
                    "intensity": intensity,
                }
            )
        print(f"[INFO] Loaded {len(lidar_frames)} LiDAR frames from {lidar_dir}")
    else:
        print(f"[WARN] LiDAR dir not found: {lidar_dir}")

    # 2) LaserScan frames (scan_2d/*.npz)
    scan_frames = []
    if scan_dir.is_dir():
        for f in sorted(scan_dir.glob("*.npz")):
            data = np.load(f)
            ts = float(data["timestamp"])
            angle_min = float(data["angle_min"])
            angle_max = float(data["angle_max"])
            angle_increment = float(data["angle_increment"])
            range_min = float(data["range_min"])
            range_max = float(data["range_max"])
            ranges = data["ranges"].astype(np.float32)

            if "intensities" in data.files:
                intensities = data["intensities"].astype(np.float32)
            else:
                intensities = None

            scan_frames.append(
                {
                    "path": f,
                    "timestamp": ts,
                    "angle_min": angle_min,
                    "angle_max": angle_max,
                    "angle_increment": angle_increment,
                    "range_min": range_min,
                    "range_max": range_max,
                    "ranges": ranges,
                    "intensities": intensities,
                }
            )
        print(f"[INFO] Loaded {len(scan_frames)} LaserScan frames from {scan_dir}")
    else:
        print(f"[WARN] LaserScan dir not found: {scan_dir}")

    # 3) IMU (imu/livox_imu.csv)
    imu_data = None
    imu_csv = imu_dir / "livox_imu.csv"
    if imu_csv.is_file():
        arr = np.loadtxt(imu_csv, delimiter=",", skiprows=1)
        arr = _ensure_2d(arr)
        # col: t,qx,qy,qz,qw,wx,wy,wz,ax,ay,az
        imu_data = {
            "t": arr[:, 0],
            "quat": arr[:, 1:5],         # (N,4)
            "ang_vel": arr[:, 5:8],      # (N,3)
            "lin_acc": arr[:, 8:11],     # (N,3)
            "raw": arr,
        }
        print(f"[INFO] Loaded IMU data: {arr.shape[0]} rows from {imu_csv}")
    else:
        print(f"[WARN] IMU csv not found: {imu_csv}")

    # 4) Pose (pose/pose_stamped.csv)
    pose_data = None
    pose_csv = pose_dir / "pose_stamped.csv"
    if pose_csv.is_file():
        arr = np.loadtxt(pose_csv, delimiter=",", skiprows=1)
        arr = _ensure_2d(arr)
        # col: t,px,py,pz,qx,qy,qz,qw
        pose_data = {
            "t": arr[:, 0],
            "pos": arr[:, 1:4],         # (N,3)
            "quat": arr[:, 4:8],        # (N,4)
            "raw": arr,
        }
        print(f"[INFO] Loaded Pose data: {arr.shape[0]} rows from {pose_csv}")
    else:
        print(f"[WARN] Pose csv not foundscan_frames: {pose_csv}")

    # 5) Odometry (odom/odom.csv)
    odom_data = None
    odom_csv = odom_dir / "odom.csv"
    if odom_csv.is_file():
        arr = np.loadtxt(odom_csv, delimiter=",", skiprows=1)
        arr = _ensure_2d(arr)
        # col: t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz
        odom_data = {
            "t": arr[:, 0],
            "pos": arr[:, 1:4],
            "quat": arr[:, 4:8],
            "lin_vel": arr[:, 8:11],
            "ang_vel": arr[:, 11:14],
            "raw": arr,
        }
        print(f"[INFO] Loaded Odom data: {arr.shape[0]} rows from {odom_csv}")
    else:
        print(f"[WARN] Odom csv not found: {odom_csv}")

    # 6) TF (tf/tf_messages.npz)
    tf_data = None
    tf_npz = tf_dir / "tf_messages.npz"
    if tf_npz.is_file():
        data = np.load(tf_npz, allow_pickle=True)
        tf_arr = data["tf"]    # dtype=object, shape (M, 11)
        # each row: [tt, topic, parent, child, tx, ty, tz, qx, qy, qz, qw]
        tf_list = []
        for row in tf_arr:
            tt, topic, parent, child, tx, ty, tz, qx, qy, qz, qw = row
            tf_list.append(
                {
                    "t": float(tt),
                    "topic": str(topic),
                    "parent": str(parent),
                    "child": str(child),
                    "translation": np.array([tx, ty, tz], dtype=float),
                    "quat": np.array([qx, qy, qz, qw], dtype=float),
                }
            )
        tf_data = {
            "list": tf_list,
            "raw": tf_arr,
        }
        print(f"[INFO] Loaded TF data: {len(tf_list)} transforms from {tf_npz}")
    else:
        print(f"[WARN] TF npz not found: {tf_npz}")


    # parsed data to dict.
    return {
        "parsed_dir": parsed_dir,
        "lidar_frames": lidar_frames,
        "scan_frames": scan_frames,
        "imu": imu_data,
        "pose": pose_data,
        "odom": odom_data,
        "tf": tf_data,
    }


def resampling():
    return




# Execution Part
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_dir",
        type=str,
        required=True,
        help="Scenario directory.",
    )
    args = parser.parse_args()
    data_raw = Read_parsed_data( args.scenario_dir )


    res = resampling( data_raw ) # resampling along timestamps