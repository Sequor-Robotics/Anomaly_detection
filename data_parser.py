from pathlib import Path
from rosbags.highlevel import AnyReader

from tqdm import tqdm
import numpy as np
import argparse
import os

# PointCloud2 → numpy Conversion Helper
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


def pointcloud2_to_struct_array(msg):
    """
    sensor_msgs/msg/PointCloud2 (ROS message) → numpy structured array

    - msg.fields : Field Layout (x, y, z, intensity, ...)
    - msg.data   : Data Byte Buffer

    return: arr["x"], arr["y"], arr["z"], arr["intensity"], ... (Structured Array)
    """
    is_bigendian = bool(msg.is_bigendian) # big-endian or little-endian

    names = []
    formats = []
    offsets = []

    # Read message data field by field
    for field in msg.fields:
        base_dtype = np.dtype( POINT_FIELD_DATATYPE_TO_DTYPE[field.datatype] )

        # Check Byte order
        if is_bigendian:
            base_dtype = base_dtype.newbyteorder(">")
        else:
            base_dtype = base_dtype.newbyteorder("<")

        count = getattr(field, "count", 1) or 1

        if count not in (0, 1):
            # vector / array
            formats.append((base_dtype, count))
        else:
            # scalar
            formats.append(base_dtype)

        names.append(field.name)
        offsets.append(field.offset)

    # Define datatype for each LiDAR point
    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": msg.point_step,  # total byte per point
    })

    num_points = msg.width * msg.height
    arr = np.frombuffer(msg.data, dtype=dtype, count=num_points)

    return arr

def extract_xyz_intensity(struct_arr):
    """
    structured array → (xyz, intensity)

    return:
    xyz: (N, 3)
    intensity: (N,) or None
    """

    xyz = np.stack(
        [struct_arr["x"], struct_arr["y"], struct_arr["z"]],
        axis=-1,
    )

    if "intensity" in struct_arr.dtype.names:
        intensity = struct_arr["intensity"]
    else:
        intensity = None

    return xyz, intensity

def parse_one_dataset(src_dir: Path):
    """
    src_dir: scenario folder containing one '.mcap' raw data file from ROS bag
    """

    src_dir = src_dir.resolve()  # as absolute path
    save_dir = src_dir / f"{src_dir.name}_parsed"

    # Output Directory
    lidar_dir = save_dir / "livox_lidar"
    laserscan_dir = save_dir / "scan_2d"
    imu_dir = save_dir / "imu"
    odom_dir = save_dir / "odom"
    pose_dir = save_dir / "pose"
    tf_dir = save_dir / "tf"

    for d in [lidar_dir, laserscan_dir, imu_dir, odom_dir, pose_dir, tf_dir]:
        os.makedirs(d, exist_ok=True)

    # Find .mcap file
    mcap_files = list(src_dir.glob("*.mcap"))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap file found in {src_dir}")
    mcap_path = mcap_files[0]  # [NOTE] make sure each folder includes only single .mcap file 
    print(f"Using mcap file: {mcap_path}")

    lidar_idx = 0
    scan_idx = 0

    imu_log = []    # each row: [t, qx, qy, qz, qw, wx, wy, wz, ax, ay, az]
    odom_log = []   # each row: [t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
    pose_log = []   # each row: [t, px, py, pz, qx, qy, qz, qw]
    tf_log = []

    # Read .mcap
    with AnyReader([mcap_path]) as reader:
        for conn, _, raw in tqdm(reader.messages(), desc="Processing messages"):
            # conn : message's topic, msgtype
            #   _  : meta-data sucah as timestaps (not required right now)
            # raw  : serialized message byte

            msg = reader.deserialize(raw, conn.msgtype)

            topic   = conn.topic
            msgtype = conn.msgtype

            header = getattr(msg, "header", None)
            if header is not None:
                t = header.stamp.sec + header.stamp.nanosec * 1e-9
            else:
                t = None

            # 1) 3D LiDAR: /livox/lidar (PointCloud2)
            if topic == "/livox/lidar" and msgtype == "sensor_msgs/msg/PointCloud2":

                struct_arr = pointcloud2_to_struct_array(msg)
                xyz, intensity = extract_xyz_intensity(struct_arr)

                # Eliminate Trash Data ( NaN / inf )
                mask = np.isfinite(xyz).all(axis=1)
                xyz = xyz[mask]
                if intensity is not None:
                    intensity = intensity[mask]

                out_path = lidar_dir / f"{lidar_idx:06d}.npz"
                if intensity is not None:
                    np.savez(
                        out_path,
                        xyz=xyz,
                        intensity=intensity,
                        timestamp=t,
                    )
                else:
                    np.savez(
                        out_path,
                        xyz=xyz,
                        timestamp=t,
                    )

                lidar_idx += 1

            # 2) 2D LaserScan: /scan (sensor_msgs/msg/LaserScan)
            elif topic == "/scan" and msgtype == "sensor_msgs/msg/LaserScan":

                # LaserScan:
                # - angle_min, angle_max
                # - angle_increment
                # - range_min, range_max
                # - ranges: float32[] (distance measured by each beam)
                # - intensities: float32[] (Optional)

                angle_min = msg.angle_min
                angle_max = msg.angle_max
                angle_increment = msg.angle_increment
                range_min = msg.range_min
                range_max = msg.range_max

                ranges = np.array(msg.ranges, dtype=np.float32)
                intensities = np.array(msg.intensities, dtype=np.float32) \
                    if len(msg.intensities) > 0 else None

                out_path = laserscan_dir / f"{scan_idx:06d}.npz"
                if intensities is not None:
                    np.savez(
                        out_path,
                        timestamp=t,
                        angle_min=angle_min,
                        angle_max=angle_max,
                        angle_increment=angle_increment,
                        range_min=range_min,
                        range_max=range_max,
                        ranges=ranges,
                        intensities=intensities,
                    )
                else:
                    np.savez(
                        out_path,
                        timestamp=t,
                        angle_min=angle_min,
                        angle_max=angle_max,
                        angle_increment=angle_increment,
                        range_min=range_min,
                        range_max=range_max,
                        ranges=ranges,
                    )
                scan_idx += 1

            # 3) IMU: /livox/imu (sensor_msgs/msg/Imu)
            elif topic == "/livox/imu" and msgtype == "sensor_msgs/msg/Imu":

                # orientation
                qx = msg.orientation.x
                qy = msg.orientation.y
                qz = msg.orientation.z
                qw = msg.orientation.w

                # angular velocity (rad/s)
                wx = msg.angular_velocity.x
                wy = msg.angular_velocity.y
                wz = msg.angular_velocity.z

                # linear acceleration (m/s^2)
                ax = msg.linear_acceleration.x
                ay = msg.linear_acceleration.y
                az = msg.linear_acceleration.z

                imu_log.append(
                    [t, qx, qy, qz, qw, wx, wy, wz, ax, ay, az]
                )

            # 4) Pose: /dlio/odom_node/pose (geometry_msgs/msg/PoseStamped)
            elif topic == "/dlio/odom_node/pose" and msgtype == "geometry_msgs/msg/PoseStamped":

                px = msg.pose.position.x
                py = msg.pose.position.y
                pz = msg.pose.position.z

                qx = msg.pose.orientation.x
                qy = msg.pose.orientation.y
                qz = msg.pose.orientation.z
                qw = msg.pose.orientation.w

                pose_log.append(
                    [t, px, py, pz, qx, qy, qz, qw]
                )

            # 5) Odom: /dlio/odom_node/odom (nav_msgs/msg/Odometry)
            elif topic == "/dlio/odom_node/odom" and msgtype == "nav_msgs/msg/Odometry":

                # pose
                px = msg.pose.pose.position.x
                py = msg.pose.pose.position.y
                pz = msg.pose.pose.position.z

                qx = msg.pose.pose.orientation.x
                qy = msg.pose.pose.orientation.y
                qz = msg.pose.pose.orientation.z
                qw = msg.pose.pose.orientation.w

                # twist (velocity)
                vx = msg.twist.twist.linear.x
                vy = msg.twist.twist.linear.y
                vz = msg.twist.twist.linear.z

                wx = msg.twist.twist.angular.x
                wy = msg.twist.twist.angular.y
                wz = msg.twist.twist.angular.z

                odom_log.append(
                    [t, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
                )

            # 6) TF / TF_STATIC (tf2_msgs/msg/TFMessage)
            elif topic in ("/tf", "/tf_static") and msgtype == "tf2_msgs/msg/TFMessage":
                for tr in msg.transforms:
                    tt = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                    parent = tr.header.frame_id
                    child = tr.child_frame_id
                    tx = tr.transform.translation.x
                    ty = tr.transform.translation.y
                    tz = tr.transform.translation.z
                    qx = tr.transform.rotation.x
                    qy = tr.transform.rotation.y
                    qz = tr.transform.rotation.z
                    qw = tr.transform.rotation.w

                    tf_log.append(
                        [tt, topic, parent, child, tx, ty, tz, qx, qy, qz, qw]
                    )

    ### Save to Local File

    # IMU
    if imu_log:
        imu_arr = np.asarray(imu_log, dtype=np.float64)
        header = "t,qx,qy,qz,qw,wx,wy,wz,ax,ay,az"
        np.savetxt(imu_dir / "livox_imu.csv", imu_arr, delimiter=",", header=header, comments="")
        print(f"IMU log saved to {imu_dir / 'livox_imu.csv'}")
    else:
        print("No IMU messages (or topic mismatch).")

    # Pose
    if pose_log:
        pose_arr = np.asarray(pose_log, dtype=np.float64)
        header = "t,px,py,pz,qx,qy,qz,qw"
        np.savetxt(pose_dir / "pose_stamped.csv", pose_arr, delimiter=",", header=header, comments="")
        print(f"Pose log saved to {pose_dir / 'pose_stamped.csv'}")
    else:
        print("No PoseStamped messages (or topic mismatch).")

    # Odom
    if odom_log:
        odom_arr = np.asarray(odom_log, dtype=np.float64)
        header = "t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz"
        np.savetxt(odom_dir / "odom.csv", odom_arr, delimiter=",", header=header, comments="")
        print(f"Odom log saved to {odom_dir / 'odom.csv'}")
    else:
        print("No Odometry messages (or topic mismatch).")

    # TF
    if tf_log:
        tf_arr = np.array(tf_log, dtype=object)
        np.savez(tf_dir / "tf_messages.npz", tf=tf_arr)
        print(f"TF log saved to {tf_dir / 'tf_messages.npz'}")
    else:
        print("No TF messages (or topic mismatch).")


def find_single_mcap_dir(d: Path):
    # check if given directory d contains only one '.mcap'

    mcap_files = list(d.glob("*.mcap"))
    if len(mcap_files) == 1:
        return True
    elif len(mcap_files) > 1:
        print(f"[WARN] {d} has more than 1 '.mcap' files ({len(mcap_files)})")
        return False
    else:
        return False


def main(args):

    src_root = Path(args.src_dir).resolve()
    print(f"\n[INFO] root src_dir: {src_root}")

    # Inspect all sub-folders in root directory (src_root)
    print(f"\n[INFO] Scanning sub-directories of {src_root} ...")
    found_any = False
    for child in sorted(src_root.iterdir()):
        if not child.is_dir():
            continue
        if find_single_mcap_dir(child):
            found_any = True
            print(f"\n[INFO] Found dataset dir: {child}")
            parse_one_dataset(child)

    if not found_any:
        print("[WARN] No subdirectories with exactly one .mcap found.")




# Execution Part
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Root directory. All subfolders with exactly one '.mcap' will be parsed.",
    )
    args = parser.parse_args()
    main(args)