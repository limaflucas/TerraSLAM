# Full controller with robust backup/resume logic + GOAL CHAINING + 7m goal tolerance
import math
import random
from controller import Robot

# ---------- Tunable parameters ----------
MAX_SPEED = 10.0
MAX_SENSOR_NUMBER = 16
MAX_SENSOR_VALUE = 1024.0
MIN_DISTANCE = 1.6   # m: start avoidance when obstacle < this

BASE_SPEED = 4.0
K_HEADING = 2.0
K_AVOID = 12.0
AVOID_MAX = 0.8
HEADING_MIN_WEIGHT = 0.45

SLOWDOWN_DIST = 3.0
GOAL_TOLERANCE = 7.0   # <<< goal reached when closer than 7m

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
    SensorData([150, 0]), SensorData([200, 0]), SensorData([300, 0]), SensorData([600, 0]),
    SensorData([0, 600]), SensorData([0, 300]), SensorData([0, 200]), SensorData([0, 150]),
] + [SensorData([0,0])]*8

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
    return sorted(goals, key=lambda g: math.hypot(g["position"][0]-current_xy[0], g["position"][1]-current_xy[1]))

# ---------- controller ----------
def main():
    robot = Robot()
    time_step = int(robot.getBasicTimeStep())
    steps_per_sec = 1000.0 / time_step

    STUCK_WINDOW_STEPS = max(1, int(STUCK_WINDOW_S * steps_per_sec))
    NAVIGATE_BACKUP_GRACE_STEPS = max(1, int(NAVIGATE_BACKUP_GRACE_S * steps_per_sec))
    BACKUP_MIN_STEPS = max(1, int(BACKUP_MIN_TIME_S * steps_per_sec))
    BACKUP_COOLDOWN_STEPS = max(1, int(BACKUP_COOLDOWN_S * steps_per_sec))

    # devices
    left_wheel = robot.getDevice("left wheel")
    right_wheel = robot.getDevice("right wheel")
    left_wheel.setPosition(float("inf"))
    right_wheel.setPosition(float("inf"))
    left_wheel.setVelocity(0.0)
    right_wheel.setVelocity(0.0)

    # LEDs
    try:
        red_leds = [robot.getDevice("red led 1"), robot.getDevice("red led 2"), robot.getDevice("red led 3")]
        for led in red_leds:
            try: led.set(0)
            except: pass
    except: red_leds = []

    for i in range(MAX_SENSOR_NUMBER):
        try:
            sensors[i].device = robot.getDevice(f"so{i}")
            sensors[i].device.enable(time_step)
        except:
            sensors[i].device = None

    try:
        lidar = robot.getDevice("RPlidar A2")
        lidar.enable(time_step)
        if hasattr(lidar, "enablePointCloud"):
            lidar.enablePointCloud()
        print("LiDAR enabled:", lidar.getName())
    except:
        lidar = None

    try:
        cam = robot.getDevice("camera")
        cam.enable(time_step)
        print("Camera enabled:", cam.getName())
    except:
        cam = None

    gps = robot.getDevice("gps"); gps.enable(time_step)
    compass = robot.getDevice("compass"); compass.enable(time_step)
    get_position = make_gps_position_provider(gps)

    # ---------- GOAL CHAINING SETUP ----------
    current_xy = get_position()
    ordered_landmarks = sort_landmarks_by_distance(current_xy, landmarks)
    manual = next((l for l in landmarks if l["name"] == manual_goal), None)
    if manual:
        ordered_landmarks = [manual] + [l for l in ordered_landmarks if l["name"] != manual_goal]

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

    while robot.step(time_step) != -1:
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
            print(f"[GOAL] reached {ordered_landmarks[goal_index]['name']} at dist={dist_to_goal:.2f}")
            goal_index += 1
            if goal_index < len(ordered_landmarks):
                remaining = ordered_landmarks[goal_index:]
                remaining = sort_landmarks_by_distance(rc, remaining)
                ordered_landmarks = ordered_landmarks[:goal_index] + remaining
                print(f"[CHAIN] next goal={ordered_landmarks[goal_index]['name']} at {ordered_landmarks[goal_index]['position']}")
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

            made_progress_toward_goal = (dist_to_goal < backup_start_dist - DIST_CHANGE_EPS)
            can_resume_due_to_time = (step_count - backup_start_step) >= BACKUP_MIN_STEPS

            can_resume = False
            if can_resume_due_to_time:
                if blen >= BACKUP_DIST:
                    can_resume = True
                elif made_progress_toward_goal:
                    can_resume = True
                elif (min_obs_dist >= MIN_DISTANCE and moved_dist > BACKUP_MIN_MOVE):
                    can_resume = True

            if can_resume:
                last_backup_step = step_count
                backup_start_pos = None
                print(f"[BACKUP->NAV] resumed after blen={blen:.2f}, moved={moved_dist:.3f}, min_obs={min_obs_dist:.2f}")
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

            last_avoid_raw = (1.0 - AVOID_FILTER_ALPHA) * last_avoid_raw + AVOID_FILTER_ALPHA * avoid_raw
            heading_supp = max(HEADING_MIN_WEIGHT, min(1.0, min_obs_dist / MIN_DISTANCE))
            heading_error = wrap_to_pi(heading_target - yaw)
            steer_goal = max(-1.0, min(1.0, K_HEADING * heading_error))
            avoid_component = max(-AVOID_MAX, min(AVOID_MAX, K_AVOID * last_avoid_raw))
            steer = avoid_component + steer_goal * heading_supp
            steer = 0.5 * last_steer + 0.5 * steer
            steer = max(-1.0, min(1.0, steer))
            last_steer = steer

            near_goal_scale = min(1.0, dist_to_goal / SLOWDOWN_DIST)
            obs_scale = 1.0 if min_obs_dist >= MIN_DISTANCE else max(0.25, min_obs_dist / MIN_DISTANCE)
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
            if same_steps >= SAME_STEPS_THRESHOLD and stuck_movement_small and (step_count >= NAVIGATE_BACKUP_GRACE_STEPS) and cooldown_ok:
                print(f"[NAV->BACKUP] same_steps={same_steps}, moved={moved_dist:.3f} -> backing up")
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
                print(f"[NAV] step={step_count} pos=({rc[0]:.3f},{rc[1]:.3f}) dist={dist_to_goal:.2f} min_obs={min_obs_dist:.2f} steer={steer:.3f} v={v_cmd:.2f} same={same_steps}")

        # LED twinkle
        delay += 1
        if delay >= DELAY:
            delay = 0
            try:
                red_leds[0].set(1)
                red_leds[0].set(0)
            except: pass

        step_count += 1


if __name__ == "__main__":
    main()

