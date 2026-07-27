# Offline RGB-D Visual SLAM with EKF

An offline visual SLAM implementation for an Ackermann-steered ROS 2 vehicle.
The pipeline reads a ROS 2 MCAP bag sequentially, predicts vehicle motion from
wheel encoders, extracts RGB-D ORB observations, and applies EKF corrections to
the robot pose and a bounded local landmark map.

The implementation is intentionally kept readable and experimental. It is a
reference for studying the individual prediction, association, mapping, and
localization steps before moving them into a real-time ROS 2 node.

## Features

- Sequential offline processing with `rosbag2_py`
- Calibrated Ackermann encoder prediction
- SE(3) robot pose with a 6×6 error-state covariance
- Depth-valid ORB feature extraction
- RGB-D back-projection into optical-frame XYZ points
- Descriptor matching with one-to-one landmark assignment
- 0.5 m geometric residual gating
- Bounded local landmark map with stale-landmark removal
- Sparse landmark observation Jacobian
- Robot and landmark EKF updates
- Wheel-odometry, visual-update, and Gazebo ground-truth comparison
- Optional 2D and 3D discarded-landmark visualization

The bag contains IMU messages, but the current estimator does not fuse the IMU.

## Dataset

The bags are stored separately because they are too large for Git:

[Download `bags.zip` from Google Drive](https://drive.google.com/file/d/1bUmqIlPAR-BLJJAN7G_Df6aMoNcbBgPu/view?usp=sharing)

Download the archive into the project root and extract it:

```bash
unzip bags.zip
```

The resulting layout should be:

```text
vSLAM_project/
├── bags/
│   ├── slam_dataset_01/
│   │   ├── metadata.yaml
│   │   └── slam_dataset_01_0.mcap
│   └── rosbag2_2026_06_02-20_52_47/
├── visual_inertial_slam/
│   ├── landmark.py
│   └── visual_slam.py
└── README.md
```

`visual_slam.py` currently processes `bags/slam_dataset_01`.

## Recorded topics

The estimator uses:

```text
/joint_states
/depth_camera/image
/depth_camera/depth_image
/depth_camera/camera_info
/ground_truth/odom
```

The bag also contains:

```text
/imu/data
```

## Requirements

- Ubuntu with ROS 2 Jazzy
- Python 3
- OpenCV
- NumPy
- SciPy
- Matplotlib
- ROS 2 MCAP storage plugin
- `cv_bridge`

On a ROS 2 Jazzy installation:

```bash
sudo apt install \
  ros-jazzy-cv-bridge \
  ros-jazzy-rosbag2-storage-mcap \
  python3-opencv \
  python3-numpy \
  python3-scipy \
  python3-matplotlib
```

## Run

```bash
cd ~/Desktop/vSLAM_project
source /opt/ros/jazzy/setup.bash
python3 visual_inertial_slam/visual_slam.py
```

The full bag is processed offline. Runtime depends on the number of active
landmarks and accepted observations.

## Output

At completion, the script displays and saves:

```text
odometry_path.png
```

The 2D figure compares:

- Wheel/Ackermann odometry
- Gazebo ground truth
- Robot pose before the visual update
- Robot pose after the visual update
- Discarded landmark positions

Set the following flag in `visual_slam.py` to enable the separate 3D landmark
figure:

```python
PLOT_LANDMARKS_3D = True
```

## Pipeline

```text
Wheel joint positions
    → Ackermann motion prediction
    → robot pose and covariance prediction

Synchronized RGB + depth
    → ORB extraction
    → optical-frame XYZ observations
    → descriptor matching
    → geometric gating
    → robot EKF update
    → landmark EKF update
    → stale-landmark removal
    → new-landmark initialization
```

## Important parameters

The current values are calibrated for the simulated vehicle:

```python
WHEEL_RADIUS = 0.096
WHEEL_BASE = 0.624
```

Other useful tuning locations include:

- ORB feature count
- Maximum active landmarks
- Maximum new landmarks per frame
- Descriptor distance and ratio thresholds
- Geometric gating threshold
- Robot process noise
- RGB-D measurement covariance
- Stale-landmark age

## Current limitations

- The parameters are tuned for this Gazebo vehicle and camera.
- Robot and landmark updates are alternated rather than maintained as one full
  joint EKF-SLAM state.
- Landmark uncertainty is simplified.
- The estimator currently has no loop closure.
- IMU measurements are recorded but not fused.
- The implementation is offline and intended primarily for learning and
  experimentation.

The ROS workspace version of this work explores a real-time keyframe front end,
keyframe-only map updates, and GTSAM pose-graph loop closure.
