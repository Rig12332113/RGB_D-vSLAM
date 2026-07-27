import rosbag2_py
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from scipy.linalg import expm

from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo, JointState, Imu
from nav_msgs.msg import Odometry

import landmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAG_PATH = str(PROJECT_ROOT / "bags" / "slam_dataset_01")
WHEEL_RADIUS = 0.096
WHEEL_BASE = 0.624
PLOT_LANDMARKS_3D = False
x = 0
y = 0
yaw = 0
previous_t = -1
previous_left_wheel = None
previous_right_wheel = None
path = []
path.append((x, y))
gt = []
slam_prediction_path = []
slam_updated_path = []
discarded_points_world = []
pending_rgb = {}
pending_depth = {}
camera_matrix = None
processed_rgbd_frames = 0

bridge = CvBridge()
orb = cv2.ORB_create(nfeatures=300)
robot_pose = np.eye(4)
robot_covariance = np.zeros((6, 6))
landmark_map = landmark.LandmarkMap()

T_CO = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
])
T_BC = np.array([
    [1, 0, 0, 0.31],
    [0, 1, 0, 0],
    [0, 0, 1, 0.25],
    [0, 0, 0, 1],
])

process_noise_std = np.array([
    0.005,                # forward/body-x: 5 mm
    0.002,                # lateral/body-y: 2 mm
    1e-5,                 # z: nearly fixed
    1e-5,                 # roll: nearly fixed
    1e-5,                 # pitch: nearly fixed
    np.deg2rad(0.1),      # yaw: 0.1 degree
])

process_noise = np.diag(process_noise_std ** 2)


def message_timestamp_ns(message):
    return (
        message.header.stamp.sec * 1_000_000_000
        + message.header.stamp.nanosec
    )


def process_rgbd_image(rgb_image, depth_image, intrinsics, timestamp_ns):
    """Extract ORB features that have a valid corresponding depth."""
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    if descriptors is None:
        return {
            "timestamp_ns": timestamp_ns,
            "keypoints": np.empty((0, 2), dtype=np.float32),
            "descriptors": np.empty((0, 32), dtype=np.uint8),
            "points_optic": np.empty((0, 3), dtype=np.float32),
        }

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    height, width = depth_image.shape

    valid_keypoints = []
    valid_descriptors = []
    points_optic = []

    for keypoint, descriptor in zip(keypoints, descriptors):
        u_float, v_float = keypoint.pt
        u = int(round(u_float))
        v = int(round(v_float))

        if not (0 <= u < width and 0 <= v < height):
            continue

        depth = float(depth_image[v, u])
        if not np.isfinite(depth) or depth <= 0.1 or depth >= 15.0:
            continue

        x_camera = (u_float - cx) * depth / fx
        y_camera = (v_float - cy) * depth / fy

        valid_keypoints.append((u_float, v_float))
        valid_descriptors.append(descriptor)
        points_optic.append((x_camera, y_camera, depth))

    return {
        "timestamp_ns": timestamp_ns,
        "keypoints": np.asarray(valid_keypoints, dtype=np.float32).reshape(-1, 2),
        "descriptors": np.asarray(
            valid_descriptors, dtype=np.uint8
        ).reshape(-1, 32),
        "points_optic": np.asarray(
            points_optic, dtype=np.float32
        ).reshape(-1, 3),
    }

def optic2world(p_optic, robot_pose):
    return robot_pose @ T_BC @ T_CO @ p_optic

def skew(p):
    return np.array([[0, -p[2], p[1]],
                     [p[2], 0, -p[0]],
                     [-p[1], p[0], 0]])

def adjoint(T):
    R = T[0:3, 0:3]
    p = T[:3, -1]
    adjoint = np.zeros((6,6))
    adjoint[0:3, 0:3] = R
    adjoint[3:6, 3:6] = R
    adjoint[0:3, -3:] = skew(p) @ R
    return adjoint

def se3_hat(xi):
    xi = np.asarray(xi).reshape(6)

    translation = xi[:3]
    rotation = xi[3:]

    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = skew(rotation)
    xi_hat[:3, 3] = translation

    return xi_hat

def se3_exp(xi):
    return expm(se3_hat(xi))


def filter_matches_geometrically(
    landmark_map,
    matches,
    points_optic,
    robot_pose,
    maximum_error=0.5,
):
    T_BW = np.linalg.inv(robot_pose)
    T_OB = np.linalg.inv(T_BC @ T_CO)
    accepted_matches = []

    for observation_index, landmark_id, distance in matches:
        observed_point = points_optic[observation_index].reshape(3, 1)
        landmark_world = np.vstack((
            landmark_map.landmarks[landmark_id].mean,
            [[1.0]],
        ))
        point_body = T_BW @ landmark_world
        predicted_point = (T_OB @ point_body)[:3]
        error = np.linalg.norm(observed_point - predicted_point)

        if error < maximum_error:
            accepted_matches.append((
                observation_index,
                landmark_id,
                distance,
            ))

    return accepted_matches


def update_robot(landmark_map, matches, points_optic, robot_pose, robot_covariance):
    if not matches:
        return robot_pose, robot_covariance
    
    H = np.zeros((3 * len(matches), 6))
    diff = np.zeros((3 * len(matches), 1))

    for observation_number, match in enumerate(matches):
        observation_index, landmark_id, _ = match
        z = points_optic[observation_index].reshape(3, 1)

        landmark_world = np.vstack((
            landmark_map.landmarks[landmark_id].mean,
            [[1.0]],
        ))

        T_BW = np.linalg.inv(robot_pose)
        T_OB = np.linalg.inv(T_BC @ T_CO)
        point_body = T_BW @ landmark_world
        point_optic = T_OB @ point_body
        z_hat = point_optic[:3]

        H_block = T_OB[:3, :3] @ np.hstack((
            -np.eye(3),
            skew(point_body[:3, 0]),
        ))

        row = 3 * observation_number

        diff[row:row+3] = z - z_hat
        H[row:row+3, :] = H_block

    measurement_covariance = (
        np.eye(3 * len(matches)) * 0.0025
    )

    PHT = robot_covariance @ H.T

    S = (
        H @ PHT
        + measurement_covariance
    )

    K = np.linalg.solve(
        S,
        PHT.T,
    ).T

    pose_error = K @ diff
    robot_pose = robot_pose @ se3_exp(pose_error)
    robot_covariance -= K @ PHT.T
    robot_covariance = 0.5 * (robot_covariance + robot_covariance.T)

    return robot_pose, robot_covariance
    


TOPIC_TYPES = {
    "/depth_camera/image": Image,
    "/depth_camera/depth_image": Image,
    "/depth_camera/camera_info": CameraInfo,
    "/joint_states": JointState,
    "/imu/data": Imu,
    "/ground_truth/odom": Odometry,
}

reader = rosbag2_py.SequentialReader()

storage_option = rosbag2_py.StorageOptions(
    uri=BAG_PATH,
    storage_id="mcap"
)

converter_option = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr",
    output_serialization_format="cdr"
)

reader.open(
    storage_option, converter_option
)

reader.set_filter(
    rosbag2_py.StorageFilter(
        topics=list(TOPIC_TYPES.keys())
    )
)

while reader.has_next():
    topic_name, serialized_data, timestamp_ns = reader.read_next()
    message_type = TOPIC_TYPES[topic_name]
    message = deserialize_message(
        serialized_data,
        message_type
    )

    if topic_name == "/joint_states":

        timestamp = timestamp_ns * 1e-9
        positions = dict(zip(message.name, message.position))
        left_wheel = positions["base_leftBack_wheel_joint"]
        right_wheel = positions["base_rightBack_wheel_joint"]

        left_steering = positions["base_leftFront_steer_joint"]
        right_steering = positions["base_rightFront_steer_joint"]

        # print(f"\
        #     timestamp: {timestamp},\
        #     left_wheel: {left_wheel},\
        #     right_wheel: {right_wheel},\
        #     left_steering: {left_steering},\
        #     right_steering: {left_steering}"
        # )

        if previous_t < 0:
            previous_t = timestamp
            previous_left_wheel = left_wheel
            previous_right_wheel = right_wheel
            continue

        delta_t = timestamp - previous_t

        steer = (left_steering + right_steering) / 2

        delta_left = WHEEL_RADIUS * (left_wheel - previous_left_wheel)
        delta_right = WHEEL_RADIUS * (right_wheel - previous_right_wheel)
        delta_distance = (delta_left + delta_right) / 2
        delta_yaw = delta_distance / WHEEL_BASE * math.tan(steer)
        midpoint_yaw = yaw + delta_yaw / 2

        x += delta_distance * math.cos(midpoint_yaw)
        y += delta_distance * math.sin(midpoint_yaw)
        yaw += delta_yaw
        
        delta_transform = np.array([
            [
                np.cos(delta_yaw),
                -np.sin(delta_yaw),
                0.0,
                delta_distance * np.cos(delta_yaw / 2),
            ],
            [
                np.sin(delta_yaw),
                np.cos(delta_yaw),
                0.0,
                delta_distance * np.sin(delta_yaw / 2),
            ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        robot_pose = robot_pose @ delta_transform
        F = adjoint(np.linalg.inv(delta_transform))
        robot_covariance = (
            F @ robot_covariance @ F.T
            + process_noise
        )
        

        path.append((x, y))
        previous_t = timestamp
        previous_left_wheel = left_wheel
        previous_right_wheel = right_wheel

    if topic_name == "/ground_truth/odom":
        position = message.pose.pose.position

        gt.append((position.x, position.y))

    if topic_name == "/depth_camera/camera_info":
        camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)

    if topic_name == "/depth_camera/image":
        image_timestamp = message_timestamp_ns(message)
        pending_rgb[image_timestamp] = message

    if topic_name == "/depth_camera/depth_image":
        image_timestamp = message_timestamp_ns(message)
        pending_depth[image_timestamp] = message

    if camera_matrix is not None:
        ready_timestamps = sorted(
            set(pending_rgb).intersection(pending_depth)
        )

        for image_timestamp in ready_timestamps:
            rgb_message = pending_rgb.pop(image_timestamp)
            depth_message = pending_depth.pop(image_timestamp)

            rgb_image = bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding="rgb8",
            )
            depth_image = bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding="passthrough",
            )

            rgbd_frame = process_rgbd_image(
                rgb_image,
                depth_image,
                camera_matrix,
                image_timestamp,
            )

            points_optic = rgbd_frame["points_optic"]
            points_optic = np.vstack((
                points_optic.T,
                np.ones((1, points_optic.shape[0])),
            ))

            matches, unmatched_indices = landmark_map.match_landmarks(rgbd_frame["descriptors"])
            matches = filter_matches_geometrically(
                landmark_map,
                matches,
                rgbd_frame["points_optic"],
                robot_pose,
                maximum_error=0.5,
            )

            # update robot
            slam_prediction_path.append((
                robot_pose[0, 3],
                robot_pose[1, 3],
            ))
            robot_pose, robot_covariance = update_robot(
                landmark_map,
                matches,
                rgbd_frame["points_optic"],
                robot_pose,
                robot_covariance,
            )
            slam_updated_path.append((
                robot_pose[0, 3],
                robot_pose[1, 3],
            ))

            points_world = optic2world(points_optic, robot_pose)
            T_WO = robot_pose @ T_BC @ T_CO
            T_OW = np.linalg.inv(T_WO)
            # match, update landmark
            landmark_map.update_landmarks(
                matches,
                rgbd_frame["points_optic"],
                T_OW,
            )

            # Discard unmatched landmarks that have not been successfully
            # updated for more than five frames.
            discarded_points_world.extend(
                landmark_map.discard_stale_landmarks(
                    matches,
                    maximum_age=5,
                )
            )

            # unmatched, add landmarks
            available_space = (
                landmark_map.maximum_point - len(landmark_map.landmarks)
            )
            number_to_add = min(
                30,
                len(unmatched_indices),
                available_space,
            )
            covariance = np.eye(3) * 0.0025
            for index in unmatched_indices[:number_to_add]:
                mean = points_world[:3, index:index + 1]
                descriptor = rgbd_frame["descriptors"][index]
                landmark_map.create_landmark(
                    mean,
                    covariance,
                    descriptor,
                    landmark_map.current_frame,
                )

            landmark_map.current_frame += 1


path = np.array(path)
gt = np.array(gt)
slam_prediction_path = np.asarray(slam_prediction_path)
slam_updated_path = np.asarray(slam_updated_path)
discarded_points_array = (
    np.asarray(discarded_points_world)
    if discarded_points_world
    else np.empty((0, 3))
)

plt.figure(figsize=(8, 8))
plt.plot(path[:, 0], path[:, 1], label="Wheel odometry")
plt.plot(gt[:, 0], gt[:, 1], label="Ground truth")
if len(slam_prediction_path) > 0:
    plt.plot(
        slam_prediction_path[:, 0],
        slam_prediction_path[:, 1],
        linestyle="--",
        label="SLAM before visual update",
    )
if len(slam_updated_path) > 0:
    plt.plot(
        slam_updated_path[:, 0],
        slam_updated_path[:, 1],
        label="SLAM after visual update",
    )
plt.scatter(path[0, 0], path[0, 1], color="green", label="Start")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Wheel odometry vs. ground truth")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("odometry_path.png", dpi=160)

landmark_figure = plt.figure(figsize=(8, 8))
landmark_axis = landmark_figure.add_subplot(111)
if len(slam_updated_path) > 0:
    landmark_axis.plot(
        slam_updated_path[:, 0],
        slam_updated_path[:, 1],
        label="SLAM path",
    )
if len(discarded_points_array) > 0:
    landmark_axis.scatter(
        discarded_points_array[:, 0],
        discarded_points_array[:, 1],
        s=8,
        alpha=0.3,
        color="red",
        label="Discarded landmarks",
    )
landmark_axis.scatter(
    path[0, 0],
    path[0, 1],
    color="green",
    label="Start",
)
landmark_axis.set_xlabel("x [m]")
landmark_axis.set_ylabel("y [m]")
landmark_axis.set_title("SLAM path and discarded landmarks")
landmark_axis.axis("equal")
landmark_axis.grid(True)
landmark_axis.legend()
landmark_figure.tight_layout()
landmark_figure.savefig("landmarks_2d.png", dpi=160)

if PLOT_LANDMARKS_3D:
    figure_3d = plt.figure(figsize=(10, 8))
    axis_3d = figure_3d.add_subplot(111, projection="3d")
    axis_3d.plot(
        path[:, 0],
        path[:, 1],
        np.zeros(len(path)),
        label="Wheel odometry",
    )
    if len(discarded_points_array) > 0:
        axis_3d.scatter(
            discarded_points_array[:, 0],
            discarded_points_array[:, 1],
            discarded_points_array[:, 2],
            s=8,
            alpha=0.3,
            color="red",
            label="Discarded landmarks",
        )
    axis_3d.scatter(
        path[0, 0],
        path[0, 1],
        0.0,
        color="green",
        label="Start",
    )
    axis_3d.set_xlabel("x [m]")
    axis_3d.set_ylabel("y [m]")
    axis_3d.set_zlabel("z [m]")
    axis_3d.set_title("Robot path and discarded landmarks")
    axis_3d.legend()
    figure_3d.tight_layout()
    figure_3d.savefig("discarded_landmarks_3d.png", dpi=160)

plt.show()
