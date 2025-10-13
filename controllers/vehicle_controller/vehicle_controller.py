from controller import Robot
import kinematics as k
import ekf
import orb_wrapper as orbw
import matplotlib.image as mpimg
import debugger
import numpy as np

# Robot
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())
CYCLES_PER_STEP = int(1000.0 / TIME_STEP)
MAX_SPEED = 6.4

# Wheels
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float("inf"))
right_wheel.setPosition(float("inf"))
left_wheel.setVelocity(0.0)
right_wheel.setVelocity(0.0)

# GPS
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

# Camera
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# EKF-SLAM
ekf_kinematics = k.Kinematics(
    x=0.0,
    y=0.0,
    theta=0.0,
    wheel_radius=(0.19 / 2.0),
    wheel_distance=0.38,
    global_timestep=TIME_STEP,
)
ekf_instance = ekf.EKF(kinematics=ekf_kinematics)
orb_instance = orbw.ORBWrapper(
    image_height=camera.getHeight(),
    image_width=camera.getWidth(),
    focal=1.8,
)

POSE_MESSAGE_FREQ = 1
CYCLES = 0


def print_messages():
    if (CYCLES % (CYCLES_PER_STEP * POSE_MESSAGE_FREQ)) == 0.0:
        position = gps.getValues()
        gps_x = position[0]
        gps_y = position[1]

        print("\n--- SLAM Update ---")
        print(
            f"Time: {robot.getTime():.2f}s | Landmarks: {ekf_instance.landmark_number}"
        )
        print(f"GPS (GT): ({gps_x:.5f}, {gps_y:.5f})")
        print(
            f"EKF (Est): ({ekf_instance.kinematics.x:.5f}, {ekf_instance.kinematics.y:.5f}, {ekf_instance.kinematics.theta:.5f})"
        )


def print_debugger(text, value):
    if (CYCLES % (CYCLES_PER_STEP * 0.5)) == 0.0:
        print(f"---> Showing: {text} {value} <---")


while robot.step(TIME_STEP) != -1:
    lwv = left_wheel.getVelocity()
    rwv = right_wheel.getVelocity()

    ekf_instance.predict(l_velocity=lwv, r_velocity=rwv)

    c_image = camera.getImage()
    orb_result = orb_instance.compute(c_image)
    # Check if we have enought frames
    if orb_result is None:
        continue

    R, t = orb_instance.get_relative_pose(orb_result.kp_pf, orb_result.kp_cf)
    # --- Step 2: Generate Range and Bearing Measurement (z_k) ---
    # Use the triangulated 3D position derived from the recovered R and t
    z_k = orb_instance.estimate_range_bearing(orb_result.kp_pf, orb_result.kp_cf, R, t)
    current_descriptor = orb_result.desc_cf[0].flatten()

    print_debugger("z_k value", z_k)
    print_debugger("current_descriptor value", current_descriptor)

    # 2. Data Association
    landmark_index = ekf_instance.find_data_association(current_descriptor, z_k)
    # 3. Correction or Augmentation
    if landmark_index == -1:
        # NEW LANDMARK: Initialize and augment
        ekf_instance.augment_state(z_k, current_descriptor)
    # print(f"*** New Landmark Added! Total: {ekf_instance.landmark_number}")
    else:
        # KNOWN LANDMARK: Correct state and covariance
        ekf_instance.correct(z_k, landmark_index)
    # print(f"Landmark {landmark_index} associated and corrected.")

    print_messages()

    left_wheel.setVelocity(MAX_SPEED)
    right_wheel.setVelocity(MAX_SPEED)
    CYCLES += 1
