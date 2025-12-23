from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from rosbags.highlevel import AnyReader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# .mcap parsing
POINT_FIELD_DATATYPE_TO_DTYPE = {
    1: np.int8,    # INT8
    2: np.uint8,   # UINT8
    3: np.int16,   # INT16
    4: np.uint16,  # UINT16
    5: np.int32,   # INT32
    6: np.uint32,  # UINT32
    7: np.float32, # FLOAT32
    8: np.float64, # FLOAT64
}

def infer_scenario_trial(trial_dir: Path) -> Tuple[str, str]:
    """
    infer scenario/trial from folder structure
    """

    trial_dir = trial_dir.resolve()
    trial_name = trial_dir.name

    # trial_id: prefer digits, pad to 3
    if trial_name.isdigit():
        trial_id = f"{int(trial_name):03d}"
    else:
        # try extract trial digits
        m = re.search(r"(\d+)$", trial_name)
        trial_id = f"{int(m.group(1)):03d}" if m else trial_name

    scen_id = "unknown"
    if trial_dir.parent is not None:
        p = trial_dir.parent.name
        if re.match(r"^[EN]\d+$", p):
            scen_id = p

    return scen_id, trial_id

def pointcloud2_to_struct_array(msg):
    is_bigendian = bool(msg.is_bigendian)

    names, formats, offsets = [], [], []
    for field in msg.fields:
        base_dtype = np.dtype(POINT_FIELD_DATATYPE_TO_DTYPE[field.datatype])
        base_dtype = base_dtype.newbyteorder(">" if is_bigendian else "<")

        count = getattr(field, "count", 1) or 1
        if count not in (0, 1):
            formats.append((base_dtype, count))
        else:
            formats.append(base_dtype)

        names.append(field.name)
        offsets.append(field.offset)

    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": msg.point_step,
    })

    num_points = msg.width * msg.height
    arr = np.frombuffer(msg.data, dtype=dtype, count=num_points)
    return arr

def extract_xyz_intensity(struct_arr):
    xyz = np.stack([struct_arr["x"], struct_arr["y"], struct_arr["z"]], axis=-1)
    intensity = struct_arr["intensity"] if "intensity" in struct_arr.dtype.names else None
    return xyz, intensity

def parse_one_mcap_to_parsed_dir(mcap_path: Path, parsed_dir: Path, overwrite: bool = False):
    """
    Parse one .mcap file into parsed_dir with subfolders:
      livox_lidar/, scan_2d/, imu/, odom/, pose/, tf/
    """
    mcap_path = mcap_path.resolve()
    parsed_dir = parsed_dir.resolve()

    if parsed_dir.exists() and not overwrite:
        # if it looks already parsed, skip
        if (parsed_dir / "livox_lidar").is_dir() or (parsed_dir / "scan_2d").is_dir():
            print(f"[SKIP] Parsed dir exists: {parsed_dir}")
            return

    os.makedirs(parsed_dir, exist_ok=True)

    lidar_dir = parsed_dir / "livox_lidar"
    laserscan_dir = parsed_dir / "scan_2d"
    imu_dir = parsed_dir / "imu"
    odom_dir = parsed_dir / "odom"
    pose_dir = parsed_dir / "pose"
    tf_dir = parsed_dir / "tf"

    for d in [lidar_dir, laserscan_dir, imu_dir, odom_dir, pose_dir, tf_dir]:
        os.makedirs(d, exist_ok=True)

    lidar_idx = 0
    scan_idx = 0
    imu_log = []
    odom_log = []
    pose_log = []
    tf_log = []

    print(f"[INFO] Parsing mcap: {mcap_path}")
    with AnyReader([mcap_path]) as reader:
        for conn, _, raw in tqdm(reader.messages(), desc="Parsing messages", dynamic_ncols=True):
            msg = reader.deserialize(raw, conn.msgtype)
            topic = conn.topic
            msgtype = conn.msgtype

            header = getattr(msg, "header", None)
            if header is not None:
                t = header.stamp.sec + header.stamp.nanosec * 1e-9
            else:
                t = None

            # 1) 3D LiDAR
            if topic == "/livox/lidar" and msgtype == "sensor_msgs/msg/PointCloud2":
                struct_arr = pointcloud2_to_struct_array(msg)
                xyz, intensity = extract_xyz_intensity(struct_arr)

                mask = np.isfinite(xyz).all(axis=1)
                xyz = xyz[mask]
                if intensity is not None:
                    intensity = intensity[mask]

                out_path = lidar_dir / f"{lidar_idx:06d}.npz"
                if intensity is not None:
                    np.savez(out_path, xyz=xyz, intensity=intensity, timestamp=t)
                else:
                    np.savez(out_path, xyz=xyz, timestamp=t)
                lidar_idx += 1

            # 2) 2D LaserScan
            elif topic == "/scan" and msgtype == "sensor_msgs/msg/LaserScan":
                out_path = laserscan_dir / f"{scan_idx:06d}.npz"
                ranges = np.array(msg.ranges, dtype=np.float32)
                intensities = np.array(msg.intensities, dtype=np.float32) if len(msg.intensities) > 0 else None

                if intensities is not None:
                    np.savez(
                        out_path,
                        timestamp=t,
                        angle_min=float(msg.angle_min),
                        angle_max=float(msg.angle_max),
                        angle_increment=float(msg.angle_increment),
                        range_min=float(msg.range_min),
                        range_max=float(msg.range_max),
                        ranges=ranges,
                        intensities=intensities,
                    )
                else:
                    np.savez(
                        out_path,
                        timestamp=t,
                        angle_min=float(msg.angle_min),
                        angle_max=float(msg.angle_max),
                        angle_increment=float(msg.angle_increment),
                        range_min=float(msg.range_min),
                        range_max=float(msg.range_max),
                        ranges=ranges,
                    )
                scan_idx += 1

            # 3) IMU
            elif topic == "/livox/imu" and msgtype == "sensor_msgs/msg/Imu":
                qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
                wx, wy, wz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
                ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
                imu_log.append([t, qx, qy, qz, qw, wx, wy, wz, ax, ay, az])

            # 4) PoseStamped
            elif topic == "/dlio/odom_node/pose" and msgtype == "geometry_msgs/msg/PoseStamped":
                px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
                qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
                pose_log.append([t, px, py, pz, qx, qy, qz, qw])

            # 5) Odometry
            elif topic == "/dlio/odom_node/odom" and msgtype == "nav_msgs/msg/Odometry":
                px, py, pz = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
                qx, qy, qz, qw = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
                vx, vy, vz = msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z
                wx, wy, wz = msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z
                odom_log.append([t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz])

            # 6) TF
            elif topic in ("/tf", "/tf_static") and msgtype == "tf2_msgs/msg/TFMessage":
                for tr in msg.transforms:
                    tt = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    parent = tr.header.frame_id
                    child = tr.child_frame_id
                    tx, ty, tz = tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z
                    qx, qy, qz, qw = tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w
                    tf_log.append([tt, topic, parent, child, tx, ty, tz, qx, qy, qz, qw])

    # Save logs
    if imu_log:
        imu_arr = np.asarray(imu_log, dtype=np.float64)
        header = "t,qx,qy,qz,qw,wx,wy,wz,ax,ay,az"
        np.savetxt(imu_dir / "livox_imu.csv", imu_arr, delimiter=",", header=header, comments="")
    if pose_log:
        pose_arr = np.asarray(pose_log, dtype=np.float64)
        header = "t,px,py,pz,qx,qy,qz,qw"
        np.savetxt(pose_dir / "pose_stamped.csv", pose_arr, delimiter=",", header=header, comments="")
    if odom_log:
        odom_arr = np.asarray(odom_log, dtype=np.float64)
        header = "t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz"
        np.savetxt(odom_dir / "odom.csv", odom_arr, delimiter=",", header=header, comments="")
    if tf_log:
        tf_arr = np.array(tf_log, dtype=object)
        np.savez(tf_dir / "tf_messages.npz", tf=tf_arr)

    print(f"[INFO] Parsed saved: {parsed_dir}")


# Export
def export_frames_to_json_named(frames, trial_dir: Path, scenario_id: str, trial_id: str, overwrite: bool = False):
    out_path = trial_dir / f"{scenario_id}_{trial_id}.json"
    if out_path.exists() and not overwrite:
        print(f"[SKIP] JSON exists: {out_path}")
        return

    serializable_frames = []
    for fr in frames:
        angles = np.asarray(fr["laser_angles"], dtype=float)
        ranges = np.asarray(fr["laser_ranges"], dtype=float)
        angle_range_pairs = np.stack([angles, ranges], axis=1)

        frame_dict = {
            "timestamp": float(fr["timestamp"]),
            "lin_vel"  : fr["lin_vel"].tolist() if fr["lin_vel"] is not None else None,
            "lin_acc"  : fr["lin_acc"].tolist() if fr["lin_acc"] is not None else None,
            "ang_vel"  : float(fr["ang_vel"]) if fr["ang_vel"] is not None else None,
            "scan"     : angle_range_pairs.tolist(),
            "position" : fr["position"].tolist() if fr["position"] is not None else None,
            "obj_pos"  : [5.0, 5.0], # default (need further labelling for scenarios with dynamic obstacles)
        }
        serializable_frames.append(frame_dict)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"frames": serializable_frames}, f, indent=2)

    print(f"[INFO] Saved JSON: {out_path}")

# Dataset discovery
def find_trial_dirs_from_src(src: Path) -> list[Path]:
    """
    If src is:
      - file.mcap                  =>  process its parent dir
      - dir with exactly 1 *.mcap  =>  process that dir
      - dir without mcap           =>  scan one-level subdirs and take those with single *.mcap
    """
    src = src.resolve()

    if src.is_file() and src.suffix.lower() == ".mcap":
        return [src.parent]

    if src.is_dir():
        mcaps = list(src.glob("*.mcap"))
        if len(mcaps) == 1:
            return [src]

        # else scan children
        trial_dirs = []
        for child in sorted(src.iterdir()):
            if not child.is_dir():
                continue
            mc = list(child.glob("*.mcap"))
            if len(mc) == 1:
                trial_dirs.append(child)
        return trial_dirs

    raise FileNotFoundError(f"Invalid src: {src}")

def get_single_mcap(trial_dir: Path) -> Path:
    mcaps = list(trial_dir.glob("*.mcap"))
    if len(mcaps) == 0:
        raise FileNotFoundError(f"No .mcap found in {trial_dir}")
    if len(mcaps) > 1:
        raise RuntimeError(f"Multiple .mcap found in {trial_dir}: {len(mcaps)}")
    return mcaps[0]