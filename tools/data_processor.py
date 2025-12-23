import numpy as np
import tqdm
from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
import json
from typing import Callable, Literal
try:
    from scipy.interpolate import CubicSpline
except Exception:
    CubicSpline = None

InterpMethod = Literal["linear", "poly", "spline"]

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

InterpMethod = Literal["linear", "poly", "spline"]

def _dedupe_sorted_time_series(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate timestamps (keep first occurrence). Assumes `t` is sorted."""
    if t.size == 0:
        return t, y
    uniq_t, first_idx = np.unique(t, return_index=True)
    order = np.argsort(first_idx)
    sel = first_idx[order]
    return t[sel], y[sel]

def polynomial_interpolation(
    t_array: np.ndarray,
    y_array: np.ndarray,
    t_star: float,
    degree: int = 3,
    window: int = 5,
):
    """Local polynomial interpolation around `t_star`."""
    t_array = np.asarray(t_array, dtype=float)
    y_array = np.asarray(y_array)

    N = t_array.shape[0]
    if N == 0:
        return None
    if N == 1:
        return y_array[0]

    # Ensure sorted by time
    if N >= 2 and np.any(np.diff(t_array) < 0):
        order = np.argsort(t_array)
        t_array = t_array[order]
        y_array = y_array[order]

    t_array, y_array = _dedupe_sorted_time_series(t_array, y_array)
    N = t_array.shape[0]
    if N == 1:
        return y_array[0]

    idx_right = np.searchsorted(t_array, t_star, side="right")
    half = window // 2
    start = max(0, idx_right - half)
    end = min(N, start + window)
    start = max(0, end - window)

    t_win = t_array[start:end]
    y_win = y_array[start:end]

    uniq_t = np.unique(t_win)
    if uniq_t.size < 2:
        return y_win[0]
    max_deg = max(1, min(degree, uniq_t.size - 1))

    if y_win.ndim == 1:
        coef = np.polyfit(t_win, y_win, deg=max_deg)
        return np.polyval(coef, t_star)

    D = y_win.shape[1]
    out = np.empty((D,), dtype=float)
    for d in range(D):
        coef = np.polyfit(t_win, y_win[:, d], deg=max_deg)
        out[d] = np.polyval(coef, t_star)
    return out

def spline_interpolation(t_array: np.ndarray, y_array: np.ndarray, t_star: float):
    """Cubic spline interpolation using SciPy if available."""
    if CubicSpline is None:
        raise RuntimeError("SciPy is required for spline interpolation but is not available.")

    t_array = np.asarray(t_array, dtype=float)
    y_array = np.asarray(y_array)

    N = t_array.shape[0]
    if N == 0:
        return None
    if N == 1:
        return y_array[0]

    if np.any(np.diff(t_array) < 0):
        order = np.argsort(t_array)
        t_array = t_array[order]
        y_array = y_array[order]

    t_array, y_array = _dedupe_sorted_time_series(t_array, y_array)
    if t_array.shape[0] == 1:
        return y_array[0]

    cs = CubicSpline(t_array, y_array, axis=0, extrapolate=True)
    return cs(t_star)

def interpolate_time_series(
    t_array: np.ndarray,
    y_array: np.ndarray,
    t_star: float,
    method: InterpMethod = "linear",
):
    method = str(method).lower().strip()
    if method not in ("linear", "poly", "spline"):
        raise ValueError(f"Unknown interpolation method: {method}. Use one of: linear|poly|spline")

    if method == "poly":
        return polynomial_interpolation(t_array, y_array, t_star)

    if method == "spline":
        return spline_interpolation(t_array, y_array, t_star)

    # ---- 기존 linear 로직 (원래 코드 내용 유지) ----
    t_array = np.asarray(t_array)
    y_array = np.asarray(y_array)

    # Find the closest time index. (= right side)
    idx_r = np.searchsorted(t_array, t_star, side="right")

    # If timestamp is outside the array range, return None
    if idx_r == 0:
        return y_array[0]
    if idx_r >= len(t_array):
        return y_array[-1]

    idx_l = idx_r - 1

    # interpolation
    t1 = t_array[idx_l]
    t2 = t_array[idx_r]
    y1 = y_array[idx_l]
    y2 = y_array[idx_r]

    # guard (avoid div by zero)
    if t2 == t1:
        return y1

    y_star = y1 + (y2 - y1) / (t2 - t1) * (t_star - t1)
    return y_star

def make_interpolator(
    t_array: np.ndarray,
    y_array: np.ndarray,
    method: InterpMethod,
) -> Callable[[float], np.ndarray | float | None]:
    """Create a callable interpolator for repeated queries.

    For spline, builds a CubicSpline once. For linear/poly, dispatches per-query.
    """
    method = str(method).lower().strip()

    if method == "spline":
        if CubicSpline is None:
            raise RuntimeError("SciPy is required for spline interpolation but is not available.")

        t = np.asarray(t_array, dtype=float)
        y = np.asarray(y_array)

        if t.size == 0:
            return lambda _ts: None
        if t.size == 1:
            v = y[0]
            return lambda _ts, _v=v: _v

        if np.any(np.diff(t) < 0):
            order = np.argsort(t)
            t = t[order]
            y = y[order]

        t, y = _dedupe_sorted_time_series(t, y)
        if t.size == 1:
            v = y[0]
            return lambda _ts, _v=v: _v

        cs = CubicSpline(t, y, axis=0, extrapolate=True)
        return lambda ts, _cs=cs: _cs(ts)

    # linear / poly
    return lambda ts: interpolate_time_series(t_array, y_array, ts, method=method)

def resampling(raw_data, method: InterpMethod):

    raw_laser = raw_data["scan_frames"]
    raw_imu   = raw_data["imu"]
    raw_odom  = raw_data["odom"]

    N_data = len(raw_laser)

    time_stamp_std  = np.array([f["timestamp"] for f in raw_laser], dtype=float)
    time_stamp_imu  = raw_imu["t"]
    time_stamp_odom = raw_odom["t"]

    imu_lin_acc = raw_imu["lin_acc"][:, :2]    # (ax, ay)
    imu_ang_vel = raw_imu["ang_vel"][:, 2]     # yaw rate -- # [NOTE] IMU

    odom_pos    = raw_odom["pos"][:, :2]       # (px, py)
    odom_linvel = raw_odom["lin_vel"][:, :2]   # (vx, vy)

    # Build interpolators once (important for spline)
    imu_lin_acc_fn = make_interpolator(time_stamp_imu, imu_lin_acc, method)
    imu_ang_vel_fn = make_interpolator(time_stamp_imu, imu_ang_vel, method)
    odom_pos_fn    = make_interpolator(time_stamp_odom, odom_pos, method)
    odom_linvel_fn = make_interpolator(time_stamp_odom, odom_linvel, method)


    resampled_frames = []

    for idx, (ts, scan_frame) in enumerate(zip(time_stamp_std, raw_laser)):

        ## Laser
        # RAW data
        angle_min = scan_frame["angle_min"]
        angle_inc = scan_frame["angle_increment"]
        raw_ranges = scan_frame["ranges"].astype(np.float64)
        raw_angles = angle_min + np.arange(raw_ranges.shape[0], dtype=np.float64) * angle_inc

        # Laser Down-Sampling
        angle_min_new = (-120) * np.pi / 180  # angle_new_min
        angle_max_new =  (120) * np.pi / 180  # angle_new_max
        N_new_samples = 128                   # number of newly down-sampled data points

        # new angles grid
        new_angles = np.linspace(angle_min_new, angle_max_new, N_new_samples)

        # nearest indices
        idx = np.searchsorted(raw_angles, new_angles)
        idx = np.clip(idx, 1, len(raw_angles) - 1)
        left = raw_angles[idx - 1]
        right = raw_angles[idx]
        choose_right = (np.abs(right - new_angles) < np.abs(left - new_angles))
        nearest_indices = idx.copy()
        nearest_indices[~choose_right] = idx[~choose_right] - 1
        new_ranges_raw = raw_ranges[nearest_indices]

        # normalization
        MAX_DIST = 5
        new_ranges = np.clip(new_ranges_raw, 0, MAX_DIST)
        new_ranges = 1 - ( new_ranges / MAX_DIST )

        # Downsampled
        angles = new_angles
        ranges = new_ranges

        # IMU
        lin_acc_star = imu_lin_acc_fn(ts)
        ang_vel_star = imu_ang_vel_fn(ts)

        # Odom
        pos_star     = odom_pos_fn(ts)
        lin_vel_star = odom_linvel_fn(ts)


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


##### Export

def load_pos_from_json(json_path: Path) -> np.ndarray:
    """
    from JSON : {"frames": [ {..., "position": [x,y], ...}, ... ]}
    """

    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    frames = obj.get("frames", None)
    if frames is None:
        raise ValueError(f"'frames' key not found in: {json_path}")

    positions = []
    for fr in frames:
        pos = fr.get("position", None)
        if pos is None: # exclude trash data
            continue
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 1 or pos.size < 2:
            continue
        pos = pos[:2]
        if not np.all(np.isfinite(pos)):
            continue
        positions.append(pos)

    if len(positions) == 0:
        return np.zeros((0, 2), dtype=float)

    return np.vstack(positions).astype(float)

def compute_total_dist(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 0.0
    diffs = np.diff(positions, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return float(seg.sum())

def save_dist(total_dist: float, trial_dir: Path, scenario_id: str, trial_id: str) -> Path:
    # {SCEN}_{TRIAL}_distance.txt
    out_path = Path(trial_dir).resolve() / f"{scenario_id}_{trial_id}_distance.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{total_dist:.6f}\n")
    print(f"[INFO] Saved distance: {out_path}")
    return out_path

def save_trj_svg(
    trial_dir: Path,
    positions: np.ndarray,
    total_dist: float,
    scenario_id: str,
    trial_id: str,
    padding_ratio: float = 0.05,
) -> Path:
    # {SCEN}_{TRIAL}_trajectory.svg
    out_path = Path(trial_dir).resolve() / f"{scenario_id}_{trial_id}_trajectory.svg"

    fig, ax = plt.subplots(figsize=(6, 6))

    if positions.shape[0] >= 2:
        xs = positions[:, 0]
        ys = positions[:, 1]

        ax.plot(xs, ys, "-", label="Trajectory")
        ax.scatter(xs[0], ys[0], s=80, marker="o", label="Start")
        ax.scatter(xs[-1], ys[-1], s=80, marker="X", label="End")

        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        x_span = x_max - x_min
        y_span = y_max - y_min
        span = max(x_span, y_span)
        if span == 0:
            span = 1.0

        pad = span * padding_ratio
        span_padded = span + 2 * pad

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0

        ax.set_xlim(x_center - span_padded / 2.0, x_center + span_padded / 2.0)
        ax.set_ylim(y_center - span_padded / 2.0, y_center + span_padded / 2.0)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.text(0.5, 0.5, "Not enough valid positions", ha="center", va="center")
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"XY Trajectory, total distance={total_dist:.4f}m")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    fig.savefig(out_path, format="svg")
    plt.close(fig)

    print(f"[INFO] Saved trajectory: {out_path}")
    return out_path
