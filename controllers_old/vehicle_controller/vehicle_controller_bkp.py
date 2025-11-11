#!/usr/bin/env python3

from controller import Robot
import kinematics as k
import ekf
import lidar_wrapper as lw
import math
import random

# --- Main Setup ---
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())
CYCLES_PER_STEP = int(1000.0 / TIME_STEP)
MAX_SPEED = 6.4

####################################
### SELF-DRIVING CONFIGURATION   ###
####################################
# ---------- Tunable parameters ----------
MAX_SPEED = 10.0
MAX_SENSOR_NUMBER = 16
MAX_SENSOR_VALUE = 1024.0
MIN_DISTANCE = 1.6  # m: start avoidance when obstacle < this

BASE_SPEED = 4.0
K_HEADING = 2.0
K_AVOID = 12.0
AVOID_MAX = 0.8
HEADING_MIN_WEIGHT = 0.45

SLOWDOWN_DIST = 3.0
GOAL_TOLERANCE = 7.0  # <<< goal reached when closer than 7m

# stuck/backing
DIST_CHANGE_EPS = 0.02
INCREASE_STEPS_THRESHOLD = 6
SAME_STEPS_THRESHOLD = 700
STUCK_WINDOW_S = 2.0
STUCK_POS_EPS = 0.3
NAVIGATE_BACKUP_GRACE_S = 3.0

BACKUP_DIST = 8.0
BACKUP_MIN_TIME_S = 0.6
BACKUP_MIN_MOVE = 0.8
BACKUP_SPEED = 4.0
BACKUP_COOLDOWN_S = 2.5

DEBUG_FREQ = 50
DELAY = 70
AVOID_FILTER_ALPHA = 0.3


# ---------- sensor & landmark setup ----------
class SensorData:
    def __init__(self, wheel_weight):
        self.device = None
        self.wheel_weight = wheel_weight


sensors = [
    SensorData([150, 0]),
    SensorData([200, 0]),
    SensorData([300, 0]),
    SensorData([600, 0]),
    SensorData([0, 600]),
    SensorData([0, 300]),
    SensorData([0, 200]),
    SensorData([0, 150]),
] + [SensorData([0, 0])] * 8

red_house = {"name": "red_house", "position": (20.0, 0.0)}
adv_board = {"name": "advertising_board", "position": (-11.54, 19.79)}
barn = {"name": "barn", "position": (-2.28, -44.03)}
ubuilding = {"name": "u_building", "position": (-47.8, -12.8)}
landmarks = [red_house, adv_board, barn, ubuilding]

manual_goal = "advertising_board"


# ---------- helpers ----------
def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def make_gps_position_provider(gps):
    def _get_position():
        v = gps.getValues()
        return (v[0], v[1])

    return _get_position


def get_heading_rad(compass):
    n = compass.getValues()
    return math.atan2(n[0], n[1])


def bearing_and_distance(current_xy, goal_xy):
    dx = goal_xy[0] - current_xy[0]
    dy = goal_xy[1] - current_xy[1]
    bearing = math.atan2(dy, dx)
    dist = math.hypot(dx, dy)
    return bearing, dist


def get_landmark_position_by_name(name):
    for l in landmarks:
        if l["name"] == name:
            return l["position"]
    return None


def sort_landmarks_by_distance(current_xy, goals):
    return sorted(
        goals,
        key=lambda g: math.hypot(
            g["position"][0] - current_xy[0], g["position"][1] - current_xy[1]
        ),
    )


# ---------- controller ----------
time_step = TIME_STEP
steps_per_sec = 1000.0 / time_step

STUCK_WINDOW_STEPS = max(1, int(STUCK_WINDOW_S * steps_per_sec))
NAVIGATE_BACKUP_GRACE_STEPS = max(1, int(NAVIGATE_BACKUP_GRACE_S * steps_per_sec))
BACKUP_MIN_STEPS = max(1, int(BACKUP_MIN_TIME_S * steps_per_sec))
BACKUP_COOLDOWN_STEPS = max(1, int(BACKUP_COOLDOWN_S * steps_per_sec))

# LEDs
try:
    red_leds = [
        robot.getDevice("red led 1"),
        robot.getDevice("red led 2"),
        robot.getDevice("red led 3"),
    ]
    for led in red_leds:
        try:
            led.set(0)
        except:
            pass
except:
    red_leds = []

for i in range(MAX_SENSOR_NUMBER):
    try:
        sensors[i].device = robot.getDevice(f"so{i}")
        sensors[i].device.enable(time_step)
    except:
        sensors[i].device = None
# Compass
compass = robot.getDevice("compass")
compass.enable(time_step)

# GPS
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)
get_position = make_gps_position_provider(gps)
# ---------- GOAL CHAINING SETUP ----------
current_xy = get_position()
ordered_landmarks = sort_landmarks_by_distance(current_xy, landmarks)
manual = next((l for l in landmarks if l["name"] == manual_goal), None)
if manual:
    ordered_landmarks = [manual] + [
        l for l in ordered_landmarks if l["name"] != manual_goal
    ]

goal_index = 0
current_goal = ordered_landmarks[goal_index]["name"]
goal_pos = ordered_landmarks[goal_index]["position"]
print(f"[INIT] Starting goal chain: {[l['name'] for l in ordered_landmarks]}")
print(f"[INIT] First goal={current_goal} at {goal_pos}")

# state
mode = "NAVIGATE"
step_count = 0
delay = 0
last_steer = 0.0
last_avoid_raw = 0.0
last_dist_to_goal = None
inc_steps = 0
same_steps = 0
recent_positions = []
backup_start_pos = None
last_backup_step = -999999
backup_start_step = -999999
backup_steer = 0.0

# Wheels
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float("inf"))
right_wheel.setPosition(float("inf"))

# Lidar
lidar = robot.getDevice("RPlidar A2")
lidar.enable(TIME_STEP)
lidar.enablePointCloud()
# Lidar Wrapper
lidar_instance = lw.LidarWrapper(min_points=3, max_points=20, max_cluster_radius=0.2)

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


# --- Main Loop ---
while robot.step(TIME_STEP) != -1:
    ########################
    ###SELF-DRIVING LOGIC###
    ########################
    rc = get_position()
    yaw = get_heading_rad(compass)
    recent_positions.append(rc)
    if len(recent_positions) > STUCK_WINDOW_STEPS:
        recent_positions.pop(0)
    moved_dist = 0.0
    if recent_positions:
        rx0, ry0 = recent_positions[0]
        moved_dist = math.hypot(rc[0] - rx0, rc[1] - ry0)
    stuck_movement_small = moved_dist < STUCK_POS_EPS

    # goal management
    if goal_index >= len(ordered_landmarks):
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        print("[DONE] all goals complete.")
        break

    goal_pos = ordered_landmarks[goal_index]["position"]
    bearing, dist_to_goal = bearing_and_distance(rc, goal_pos)

    # ---- goal reached condition (7m) ----
    if dist_to_goal <= GOAL_TOLERANCE:
        print(
            f"[GOAL] reached {ordered_landmarks[goal_index]['name']} at dist={dist_to_goal:.2f}"
        )
        goal_index += 1
        if goal_index < len(ordered_landmarks):
            remaining = ordered_landmarks[goal_index:]
            remaining = sort_landmarks_by_distance(rc, remaining)
            ordered_landmarks = ordered_landmarks[:goal_index] + remaining
            print(
                f"[CHAIN] next goal={ordered_landmarks[goal_index]['name']} at {ordered_landmarks[goal_index]['position']}"
            )
            same_steps = inc_steps = 0
            last_dist_to_goal = None
            continue
        else:
            print("[DONE] all goals reached.")
            break

        # ----- MODE: BACKUP -----
    if mode == "BACKUP":
        if backup_start_pos is None:
            backup_start_pos = rc
            backup_start_step = step_count
            backup_start_dist = dist_to_goal

        v_cmd = -BACKUP_SPEED
        steer = backup_steer
        left_speed = v_cmd * (1.0 - steer)
        right_speed = v_cmd * (1.0 + steer)
        left_wheel.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left_speed)))
        right_wheel.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right_speed)))

        bdx = rc[0] - backup_start_pos[0]
        bdy = rc[1] - backup_start_pos[1]
        blen = math.hypot(bdx, bdy)

        min_obs_dist = float("inf")
        for i in range(MAX_SENSOR_NUMBER):
            dev = sensors[i].device
            if dev is None:
                continue
            sv = dev.getValue()
            if sv <= 0.0:
                continue
            obs_d = 5.0 * (1.0 - (sv / MAX_SENSOR_VALUE))
            min_obs_dist = min(min_obs_dist, obs_d)

        made_progress_toward_goal = dist_to_goal < backup_start_dist - DIST_CHANGE_EPS
        can_resume_due_to_time = (step_count - backup_start_step) >= BACKUP_MIN_STEPS

        can_resume = False
        if can_resume_due_to_time:
            if blen >= BACKUP_DIST:
                can_resume = True
            elif made_progress_toward_goal:
                can_resume = True
            elif min_obs_dist >= MIN_DISTANCE and moved_dist > BACKUP_MIN_MOVE:
                can_resume = True

        if can_resume:
            last_backup_step = step_count
            backup_start_pos = None
            print(
                f"[BACKUP->NAV] resumed after blen={blen:.2f}, moved={moved_dist:.3f}, min_obs={min_obs_dist:.2f}"
            )
            random_nudge = random.uniform(-0.15, 0.15)
            heading_target = wrap_to_pi(bearing + random_nudge)
            mode = "NAVIGATE"
            recent_positions = []
            last_dist_to_goal = None
            inc_steps = 0
            same_steps = 0

    # ----- MODE: NAVIGATE -----
    else:
        heading_target = bearing

        left_wt = 0.0
        right_wt = 0.0
        min_obs_dist = float("inf")
        for i in range(MAX_SENSOR_NUMBER):
            dev = sensors[i].device
            if dev is None:
                continue
            sv = dev.getValue()
            if sv <= 0.0:
                continue
            obs_d = 5.0 * (1.0 - (sv / MAX_SENSOR_VALUE))
            min_obs_dist = min(min_obs_dist, obs_d)
            if obs_d < MIN_DISTANCE:
                speed_modifier = 1.0 - (obs_d / MIN_DISTANCE)
                left_wt += sensors[i].wheel_weight[0] * speed_modifier
                right_wt += sensors[i].wheel_weight[1] * speed_modifier

        avoid_raw = 0.0
        if min_obs_dist < MIN_DISTANCE:
            wt_sum = max(left_wt + right_wt, 1e-6)
            avoid_raw = (right_wt - left_wt) / wt_sum

        last_avoid_raw = (
            1.0 - AVOID_FILTER_ALPHA
        ) * last_avoid_raw + AVOID_FILTER_ALPHA * avoid_raw
        heading_supp = max(HEADING_MIN_WEIGHT, min(1.0, min_obs_dist / MIN_DISTANCE))
        heading_error = wrap_to_pi(heading_target - yaw)
        steer_goal = max(-1.0, min(1.0, K_HEADING * heading_error))
        avoid_component = max(-AVOID_MAX, min(AVOID_MAX, K_AVOID * last_avoid_raw))
        steer = avoid_component + steer_goal * heading_supp
        steer = 0.5 * last_steer + 0.5 * steer
        steer = max(-1.0, min(1.0, steer))
        last_steer = steer

        near_goal_scale = min(1.0, dist_to_goal / SLOWDOWN_DIST)
        obs_scale = (
            1.0
            if min_obs_dist >= MIN_DISTANCE
            else max(0.25, min_obs_dist / MIN_DISTANCE)
        )
        v_cmd = BASE_SPEED * near_goal_scale * obs_scale
        left_speed = v_cmd * (1.0 - steer)
        right_speed = v_cmd * (1.0 + steer)
        left_wheel.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left_speed)))
        right_wheel.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right_speed)))

        if last_dist_to_goal is None:
            last_dist_to_goal = dist_to_goal
        else:
            dd = dist_to_goal - last_dist_to_goal
            if dd > DIST_CHANGE_EPS:
                inc_steps += 1
                same_steps = 0
            elif abs(dd) <= DIST_CHANGE_EPS:
                same_steps += 1
                inc_steps = 0
            else:
                inc_steps = 0
                same_steps = 0
            last_dist_to_goal = dist_to_goal

        cooldown_ok = (step_count - last_backup_step) >= BACKUP_COOLDOWN_STEPS
        if (
            same_steps >= SAME_STEPS_THRESHOLD
            and stuck_movement_small
            and (step_count >= NAVIGATE_BACKUP_GRACE_STEPS)
            and cooldown_ok
        ):
            print(
                f"[NAV->BACKUP] same_steps={same_steps}, moved={moved_dist:.3f} -> backing up"
            )
            mode = "BACKUP"
            backup_start_pos = None
            backup_start_dist = dist_to_goal
            backup_start_step = step_count
            backup_steer = random.uniform(-0.3, 0.3)
            recent_positions = []
            last_dist_to_goal = None
            inc_steps = 0
            same_steps = 0

        if step_count % DEBUG_FREQ == 0:
            print(
                f"[NAV] step={step_count} pos=({rc[0]:.3f},{rc[1]:.3f}) dist={dist_to_goal:.2f} min_obs={min_obs_dist:.2f} steer={steer:.3f} v={v_cmd:.2f} same={same_steps}"
            )

    # LED twinkle
    delay += 1
    if delay >= DELAY:
        delay = 0
        try:
            red_leds[0].set(1)
            red_leds[0].set(0)
        except:
            pass
    step_count += 1

    # Set wheel velocities
    left_wheel.setVelocity(MAX_SPEED)
    right_wheel.setVelocity(MAX_SPEED)

    # Get wheel velocities for prediction
    lwv = left_wheel.getVelocity()
    rwv = right_wheel.getVelocity()

    # 1. EKF Prediction Step (based on kinematics)
    ekf_instance.predict(l_velocity=lwv, r_velocity=rwv)

    # 2. Lidar Measurement Step
    point_cloud = lidar.getPointCloud()
    landmarks = lidar_instance.extract_landmarks(point_cloud)

    # 3. EKF Update Step (for each observed landmark)
    for z_k in landmarks:
        # Data Association
        landmark_index = ekf_instance.find_data_association(z_k)

        # Correction or Augmentation
        if landmark_index == -1:
            # New landmark: Augment the state
            ekf_instance.augment_state(z_k)
        else:
            # Known landmark: Correct the state
            ekf_instance.correct(z_k, landmark_index)

    # Print status messages
    print_messages()
    CYCLES += 1








    /*
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <webots/motor.h>
#include <webots/robot.h>

#include <webots/camera.h>
#include <webots/distance_sensor.h>
#include <webots/led.h>

#include <stdio.h>
#include <stdlib.h>

#define MAX_SPEED 47.6

#define NUMBER_OF_ULTRASONIC_SENSORS 5
static const char *ultrasonic_sensors_names[NUMBER_OF_ULTRASONIC_SENSORS] = {
  "left ultrasonic sensor", "front left ultrasonic sensor", "front ultrasonic sensor", "front right ultrasonic sensor",
  "right ultrasonic sensor"};

#define NUMBER_OF_INFRARED_SENSORS 12
static const char *infrared_sensors_names[NUMBER_OF_INFRARED_SENSORS] = {
  // turret sensors
  "rear left infrared sensor", "left infrared sensor", "front left infrared sensor", "front infrared sensor",
  "front right infrared sensor", "right infrared sensor", "rear right infrared sensor", "rear infrared sensor",
  // ground sensors
  "ground left infrared sensor", "ground front left infrared sensor", "ground front right infrared sensor",
  "ground right infrared sensor"};

int main(int argc, char **argv) {
  wb_robot_init();

  int time_step = (int)wb_robot_get_basic_time_step();
  int i;

  // get and enable the camera
  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, time_step);

  // get and enable the ultrasonic sensors
  WbDeviceTag ultrasonic_sensors[5];
  for (i = 0; i < 5; ++i) {
    ultrasonic_sensors[i] = wb_robot_get_device(ultrasonic_sensors_names[i]);
    wb_distance_sensor_enable(ultrasonic_sensors[i], time_step);
  }

  // get and enable the infrared sensors
  WbDeviceTag infrared_sensors[12];
  for (i = 0; i < 12; ++i) {
    infrared_sensors[i] = wb_robot_get_device(infrared_sensors_names[i]);
    wb_distance_sensor_enable(infrared_sensors[i], time_step);
  }

  // get the led actuators
  WbDeviceTag leds[3] = {wb_robot_get_device("front left led"), wb_robot_get_device("front right led"),
                         wb_robot_get_device("rear led")};

  // get the motors and set target position to infinity (speed control)
  WbDeviceTag left_motor, right_motor;
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);

  // store the last time a message was displayed
  int last_display_second = 0;

  // main loop
  while (wb_robot_step(time_step) != -1) {
    // display some sensor data every second
    // and change randomly the led colors
    int display_second = (int)wb_robot_get_time();
    if (display_second != last_display_second) {
      last_display_second = display_second;

      printf("time = %d [s]\n", display_second);
      for (i = 0; i < 5; ++i)
        printf("- ultrasonic sensor('%s') = %f [m]\n", ultrasonic_sensors_names[i],
               wb_distance_sensor_get_value(ultrasonic_sensors[i]));
      for (i = 0; i < 12; ++i)
        printf("- infrared sensor('%s') = %f [m]\n", infrared_sensors_names[i],
               wb_distance_sensor_get_value(infrared_sensors[i]));

      for (i = 0; i < 3; ++i)
        wb_led_set(leds[i], 0xFFFFFF & rand());
    }

    // simple obstacle avoidance algorithm
    // based on the front infrared sensors
    double speed_offset = 0.2 * (MAX_SPEED - 0.03 * wb_distance_sensor_get_value(infrared_sensors[3]));
    double speed_delta =
      0.03 * wb_distance_sensor_get_value(infrared_sensors[2]) - 0.03 * wb_distance_sensor_get_value(infrared_sensors[4]);
    wb_motor_set_velocity(left_motor, speed_offset + speed_delta);
    wb_motor_set_velocity(right_motor, speed_offset - speed_delta);
  };

  wb_robot_cleanup();

  return EXIT_SUCCESS;
}


/*
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <string.h>

#include <webots/accelerometer.h>
#include <webots/camera.h>
#include <webots/compass.h>
#include <webots/distance_sensor.h>
#include <webots/gyro.h>
#include <webots/lidar.h>
#include <webots/light_sensor.h>
#include <webots/motor.h>
#include <webots/position_sensor.h>
#include <webots/range_finder.h>
#include <webots/robot.h>

#define TIME_STEP 32
#define MAX_VELOCITY 26

int main(int argc, char *argv[]) {
  /* define variables */
  /* motors */
  WbDeviceTag front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, front_left_position_sensor,
    front_right_position_sensor, rear_left_position_sensor, rear_right_position_sensor;
  double avoidance_speed[2];
  const double base_speed = 6.0;
  double motor_speed[2];
  /* RGBD camera */
  WbDeviceTag camera_rgb, camera_depth;
  /* rotational lidar */
  WbDeviceTag lidar;
  /* IMU */
  WbDeviceTag accelerometer, gyro, compass;
  /* distance sensors */
  WbDeviceTag distance_sensors[4];
  double distance_sensors_value[4];

  // set empirical coefficients for collision avoidance
  const double coefficients[2][2] = {{15.0, -9.0}, {-15.0, 9.0}};

  int i, j;

  /* initialize Webots */
  wb_robot_init();

  /* get a handler to the motors and set target position to infinity (speed control). */
  front_left_motor = wb_robot_get_device("fl_wheel_joint");
  front_right_motor = wb_robot_get_device("fr_wheel_joint");
  rear_left_motor = wb_robot_get_device("rl_wheel_joint");
  rear_right_motor = wb_robot_get_device("rr_wheel_joint");
  wb_motor_set_position(front_left_motor, INFINITY);
  wb_motor_set_position(front_right_motor, INFINITY);
  wb_motor_set_position(rear_left_motor, INFINITY);
  wb_motor_set_position(rear_right_motor, INFINITY);
  wb_motor_set_velocity(front_left_motor, 0.0);
  wb_motor_set_velocity(front_right_motor, 0.0);
  wb_motor_set_velocity(rear_left_motor, 0.0);
  wb_motor_set_velocity(rear_right_motor, 0.0);

  /* get a handler to the position sensors and enable them. */
  front_left_position_sensor = wb_robot_get_device("front left wheel motor sensor");
  front_right_position_sensor = wb_robot_get_device("front right wheel motor sensor");
  rear_left_position_sensor = wb_robot_get_device("rear left wheel motor sensor");
  rear_right_position_sensor = wb_robot_get_device("rear right wheel motor sensor");
  wb_position_sensor_enable(front_left_position_sensor, TIME_STEP);
  wb_position_sensor_enable(front_right_position_sensor, TIME_STEP);
  wb_position_sensor_enable(rear_left_position_sensor, TIME_STEP);
  wb_position_sensor_enable(rear_right_position_sensor, TIME_STEP);

  /* get a handler to the ASTRA rgb and depth cameras and enable them. */
  camera_rgb = wb_robot_get_device("camera rgb");
  camera_depth = wb_robot_get_device("camera depth");
  wb_camera_enable(camera_rgb, TIME_STEP);
  wb_range_finder_enable(camera_depth, TIME_STEP);

  /* get a handler to the RpLidarA2 and enable it. */
  lidar = wb_robot_get_device("laser");
  wb_lidar_enable(lidar, TIME_STEP);
  wb_lidar_enable_point_cloud(lidar);

  /* get a handler to the IMU devices and enable them. */
  accelerometer = wb_robot_get_device("imu accelerometer");
  gyro = wb_robot_get_device("imu gyro");
  compass = wb_robot_get_device("imu compass");
  wb_accelerometer_enable(accelerometer, TIME_STEP);
  wb_gyro_enable(gyro, TIME_STEP);
  wb_compass_enable(compass, TIME_STEP);

  /* get a handler to the distance sensors and enable them. */
  distance_sensors[0] = wb_robot_get_device("fl_range");
  distance_sensors[1] = wb_robot_get_device("rl_range");
  distance_sensors[2] = wb_robot_get_device("fr_range");
  distance_sensors[3] = wb_robot_get_device("rr_range");
  wb_distance_sensor_enable(distance_sensors[0], TIME_STEP);
  wb_distance_sensor_enable(distance_sensors[1], TIME_STEP);
  wb_distance_sensor_enable(distance_sensors[2], TIME_STEP);
  wb_distance_sensor_enable(distance_sensors[3], TIME_STEP);

  /* main loop */
  while (wb_robot_step(TIME_STEP) != -1) {
    /* get accelerometer values */
    const double *a = wb_accelerometer_get_values(accelerometer);
    printf("accelerometer values = %0.2f %0.2f %0.2f\n", a[0], a[1], a[2]);

    /* get distance sensors values */
    for (i = 0; i < 4; i++)
      distance_sensors_value[i] = wb_distance_sensor_get_value(distance_sensors[i]);

    /* compute motors speed */
    for (i = 0; i < 2; ++i) {
      avoidance_speed[i] = 0.0;
      for (j = 1; j < 3; ++j)
        avoidance_speed[i] += (2.0 - distance_sensors_value[j]) * (2.0 - distance_sensors_value[j]) * coefficients[i][j - 1];
      motor_speed[i] = base_speed + avoidance_speed[i];
      motor_speed[i] = motor_speed[i] > MAX_VELOCITY ? MAX_VELOCITY : motor_speed[i];
    }

    /* set speed values */
    wb_motor_set_velocity(front_left_motor, motor_speed[0]);
    wb_motor_set_velocity(front_right_motor, motor_speed[1]);
    wb_motor_set_velocity(rear_left_motor, motor_speed[0]);
    wb_motor_set_velocity(rear_right_motor, motor_speed[1]);
  }

  wb_robot_cleanup();

  return 0;
}
