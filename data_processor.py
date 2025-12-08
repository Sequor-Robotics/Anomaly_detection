import numpy as np
import tqdm
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
import json


##### Data loading

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
      - case 1) Original Scenario Folder (Ex: /expert/scene01) -> recognize /expert/scene01/scene01_parsed automatically
      - case 2) On *_parsed Folder (Ex: /expert/scene01/scene01_parsed) -> Used directly
    """

    scenario_path = Path(scenario_path).resolve()

    # detect parsed directory
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



##### RESAMPLING

def find_bracket_indices(t_array: np.ndarray, t_star: float):

    idx_right = np.searchsorted(t_array, t_star, side="right")

    idx1 = idx_right - 1
    if idx1 < 0:
        idx1 = None

    idx2 = idx_right
    if idx2 >= len(t_array):
        idx2 = None

    return idx1, idx2

def linear_interpolation(t1, y1, t2, y2, t_star):
    y_star = y1 + ( y2- y1 )/( t2 - t1 )*( t_star - t1 )
    return y_star

def interpolate_time_series(t_array: np.ndarray,
                            y_array: np.ndarray,
                            t_star: float):

    t_array = np.asarray(t_array)
    y_array = np.asarray(y_array)

    N = t_array.shape[0]
    if N == 0:
        return None

    idx1, idx2 = find_bracket_indices(t_array, t_star)

    ### Interpolation case
    if idx1 is not None and idx2 is not None:
        t1 = t_array[idx1]
        t2 = t_array[idx2]
        y1 = y_array[idx1]
        y2 = y_array[idx2]
        return linear_interpolation(t1, y1, t2, y2, t_star)

    ### Extrapolation case
    if N == 1:   # single data point (non-timeserial variable)
        return y_array[0]

    # t_star < t_array[0]
    if idx1 is None and idx2 is not None:
        t1 = t_array[0]
        t2 = t_array[1]
        y1 = y_array[0]
        y2 = y_array[1]
        return linear_interpolation(t1, y1, t2, y2, t_star)  # Use t[0], t[1]

    # t_star > t_array[-1]
    if idx1 is not None and idx2 is None:
        t1 = t_array[-2]
        t2 = t_array[-1]
        y1 = y_array[-2]
        y2 = y_array[-1]
        return linear_interpolation(t1, y1, t2, y2, t_star)  # Use t[-2], t[-1]

    return None

def resampling(raw_data):

    raw_laser = raw_data["scan_frames"]
    raw_imu   = raw_data["imu"]
    raw_odom  = raw_data["odom"]

    N_data = len(raw_laser)

    time_stamp_std  = np.array([f["timestamp"] for f in raw_laser], dtype=float)
    time_stamp_imu  = raw_imu["t"]
    time_stamp_odom = raw_odom["t"]

    imu_lin_acc = raw_imu["lin_acc"][:, :2]    # (ax, ay)
    imu_ang_vel = raw_imu["ang_vel"][:, 2]     # yaw rate -- # [NOTE] SLAM or IMU ??? ... Use better cue

    odom_pos    = raw_odom["pos"][:, :2]       # (px, py)
    odom_linvel = raw_odom["lin_vel"][:, :2]   # (vx, vy)

    resampled_frames = []

    for idx, (ts, scan_frame) in enumerate(zip(time_stamp_std, raw_laser)):

        # Laser
        # [NOTE] No need to include all the data points ... Extract some of them
        angle_min = scan_frame["angle_min"]
        angle_inc = scan_frame["angle_increment"]
        ranges    = scan_frame["ranges"].astype(np.float64)
        angles    = angle_min + np.arange(ranges.shape[0], dtype=np.float64) * angle_inc

        # IMU
        lin_acc_star = interpolate_time_series(time_stamp_imu, imu_lin_acc, ts)
        ang_vel_star = interpolate_time_series(time_stamp_imu, imu_ang_vel, ts)

        # Odom
        pos_star     = interpolate_time_series(time_stamp_odom, odom_pos, ts)
        lin_vel_star = interpolate_time_series(time_stamp_odom, odom_linvel, ts)

        frame_data = {
            "timestamp": ts,
            "lin_vel": lin_vel_star,          # shape (2,)
            "lin_acc": lin_acc_star,          # shape (2,)
            "ang_vel": ang_vel_star,          # scalar
            "laser_angles": angles,           # np.ndarray (M,)
            "laser_ranges": ranges,           # np.ndarray (M,)
            "position": pos_star,             # shape (2,)
        }

        resampled_frames.append(frame_data)

    return resampled_frames



##### Processing


# [NOTE] Lidar points .. normalization ?
# [NOTE] Lidar points .. No need to use it all. Extract some of them.
# [NOTE] Negative dataset (near obstacles) .. need to cut some data b/c robot got far away from racks while traversing corridors during data collection



##### Export

def compute_total_dist(frames):

    positions = []
    timestamps = []

    for fr in frames:
        pos = fr.get("position", None)
        if pos is None:
            continue
        pos = np.asarray(pos, dtype=float)

        # Skip NaN
        if not np.all(np.isfinite(pos)):
            continue

        positions.append(pos)
        timestamps.append(fr["timestamp"])

    positions = np.asarray(positions)       # shape (N, 2) or (N, 3)
    timestamps = np.asarray(timestamps)     # shape (N,)

    if positions.shape[0] < 2:
        return 0.0

    diffs = np.diff(positions, axis=0)               # shape (N-1, dim)
    segment_lengths = np.linalg.norm(diffs, axis=1)  # shape (N-1,)
    total_dist = float(segment_lengths.sum())

    return positions, total_dist

def create_image( positions, d, frames, scenario_dir: str | Path, save_img: bool = False):

    scenario_path = Path(scenario_dir).resolve()

    if scenario_path.name.endswith("_parsed"):
        scenario_root = scenario_path.parent
    else:
        scenario_root = scenario_path

    scenario_name = scenario_root.name
    out_path = scenario_root / f"{scenario_name}_trajectory.svg"

    if save_img and positions.shape[0] >= 2:
        xs = positions[:, 0]
        ys = positions[:, 1]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(xs, ys, "-", label="Trajectory")  # Total trajectory

        ax.scatter(xs[0], ys[0], s=80, color="green", marker="o", label="Start")  # Starting point (Green)
        ax.scatter(xs[-1], ys[-1], s=80, color="red", marker="X", label="End")    # Ending point (Red)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"XY Trajectory, total distance={d:.4f}m")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig( out_path )

def export_frames_to_json(frames, scenario_dir: str | Path, save_json: bool = False):

    scenario_path = Path(scenario_dir).resolve()

    if scenario_path.name.endswith("_parsed"):
        scenario_root = scenario_path.parent
    else:
        scenario_root = scenario_path

    scenario_name = scenario_root.name
    out_path = scenario_root / f"{scenario_name}_processed.json"

    serializable_frames = []

    for fr in frames:
        angles = np.asarray(fr["laser_angles"], dtype=float)
        ranges = np.asarray(fr["laser_ranges"], dtype=float)

        # (M, 2) .. angle–range matching
        angle_range_pairs = np.stack([angles, ranges], axis=1)

        frame_dict = {
            "timestamp": float(fr["timestamp"]),
            "lin_vel": fr["lin_vel"].tolist() if fr["lin_vel"] is not None else None,
            "lin_acc": fr["lin_acc"].tolist() if fr["lin_acc"] is not None else None,
            "ang_vel": float(fr["ang_vel"]) if fr["ang_vel"] is not None else None,
            "scan": angle_range_pairs.tolist(),  # [[angle0, range0], [angle1, range1], ...] -> LaserScan data: (N, 2) array
            "position": fr["position"].tolist() if fr["position"] is not None else None,
        }

        serializable_frames.append(frame_dict)

    if save_json:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"frames": serializable_frames},
                f,
                indent=2
            )

        print(f"[INFO] Saved JSON to {out_path.resolve()}")

def process_one_scenario(scenario_dir, save_img=False, save_json=False):

    # 1. Load parsed data
    data_raw = Read_parsed_data( scenario_dir )

    # 2. Resampling for time sync.
    res = resampling( data_raw ) # resampling along timestamps ... return data by frames

    # 3. Calculate distance travelled
    pos, total_dist = compute_total_dist( res )

    # 4. Visualize & save an image
    create_image(pos, total_dist, res, scenario_dir, save_img=save_img)

    # 5. Save to local .json file
    export_frames_to_json( res, scenario_dir, save_json=save_json)



### Execution Part
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Root directory. All subfolders of each scenario will be processed.",
    )
    args = parser.parse_args()

    src_root = Path(args.src_dir).resolve()
    print(f"[INFO] root src_dir: {src_root}")

    # Inspect all sub-folders in root directory (src_root)
    print(f"[INFO] Scanning sub-directories of {src_root} ...")
    found_any = False
    for child in sorted(src_root.iterdir()):
        if not child.is_dir():
            continue

        parsed_dirs = [
            p for p in child.iterdir()
            if p.is_dir() and p.name.endswith("_parsed")
        ]

        if not parsed_dirs:
            continue
        
        found_any = True
        print(f"\n[INFO] Found scenario dir: {child}")
        process_one_scenario(child, save_img=False, save_json=True)

    if not found_any:
        print("[WARN] No sub-directories with parsed data found.")