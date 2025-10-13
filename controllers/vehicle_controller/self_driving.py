#!/usr/bin/env python3
import math


class SelfDriving:
    def __init__(
        self, steps_per_sec=0.0, gps=None, compass=None, mode="NAVIGATE", manual_goal=[]
    ):
        self.gps = gps
        self.compass = compass
        # stuck/backing
        self.STUCK_WINDOW_S = 2.0
        self.NAVIGATE_BACKUP_GRACE_S = 3.0
        self.BACKUP_MIN_TIME_S = 0.6
        self.BACKUP_COOLDOWN_S = 2.5
        self.STUCK_POS_EPS = 0.3
        # navigation
        self.STUCK_WINDOW_STEPS = max(1, int(self.STUCK_WINDOW_S * steps_per_sec))
        self.NAVIGATE_BACKUP_GRACE_STEPS = max(
            1, int(self.NAVIGATE_BACKUP_GRACE_S * steps_per_sec)
        )
        self.BACKUP_MIN_STEPS = max(1, int(self.BACKUP_MIN_TIME_S * steps_per_sec))
        self.BACKUP_COOLDOWN_STEPS = max(1, int(self.BACKUP_COOLDOWN_S * steps_per_sec))

        # goals
        self.GOAL_TOLERANCE = 7.0  # <<< goal reached when closer than 7m
        self.manual_goal = manual_goal
        self.goal_index = 0
        self.ordered_landmarks = None
        self.current_goal = None
        self.goal_pos = None
        # states
        self.mode = mode
        self.last_steer = 0.0
        self.last_avoid_raw = 0.0
        self.last_dist_to_goal = None
        self.inc_steps = 0
        self.same_steps = 0
        self.recent_positions = []
        self.backup_start_pos = None
        self.last_backup_step = -999999
        self.backup_start_step = -999999
        self.backup_steer = 0.0

    def activate(self):
        rc = self.get_position()
        yaw = self.get_heading_rad()
        self.recent_positions.append(rc)
        if len(self.recent_positions) > self.STUCK_WINDOW_STEPS:
            self.recent_positions.pop(0)
        moved_dist = 0.0
        if self.recent_positions:
            rx0, ry0 = self.recent_positions[0]
            moved_dist = math.hypot(rc[0] - rx0, rc[1] - ry0)
        stuck_movement_small = moved_dist < self.STUCK_POS_EPS

        # goal management
        if self.goal_index >= len(self.ordered_landmarks):
            print("[DONE] all goals complete.")
            return (0, 0)

        self.goal_pos = self.ordered_landmarks[self.goal_index]["position"]
        bearing, dist_to_goal = self.bearing_and_distance(rc, self.goal_pos)

        # ---- goal reached condition (7m) ----
        if dist_to_goal <= self.GOAL_TOLERANCE:
            print(
                f"[GOAL] reached {self.ordered_landmarks[self.goal_index]['name']} at dist={dist_to_goal:.2f}"
            )
            self.goal_index += 1
            if self.goal_index < len(self.ordered_landmarks):
                remaining = self.ordered_landmarks[self.goal_index :]
                remaining = self.sort_landmarks_by_distance(rc, remaining)
                ordered_landmarks = (
                    self.ordered_landmarks[: self.goal_index] + remaining
                )
                print(
                    f"[CHAIN] next goal={ordered_landmarks[self.goal_index]['name']} at {ordered_landmarks[self.goal_index]['position']}"
                )
                same_steps = inc_steps = 0
                last_dist_to_goal = None
                continue
            else:
                print("[DONE] all goals reached.")
                break

    def goal_chaining_setup(self, landmarks):
        current_xy = self.get_position()
        self.ordered_landmarks = self.sort_landmarks_by_distance(current_xy, landmarks)
        manual = next((l for l in landmarks if l["name"] in self.manual_goal), None)
        if manual:
            self.ordered_landmarks = [manual] + [
                l for l in self.ordered_landmarks if l["name"] not in self.manual_goal
            ]
        self.current_goal = self.ordered_landmarks[self.goal_index]["name"]
        self.goal_pos = self.ordered_landmarks[self.goal_index]["position"]
        print(
            f"[INIT] Starting goal chain: {[l['name'] for l in self.ordered_landmarks]}"
        )
        print(f"[INIT] First goal={current_goal} at {goal_pos}")

    def get_position(self):
        v = self.gps.getValues()
        return (v[0], v[1])

    def get_heading_rad(self):
        n = self.compass.getValues()
        return math.atan2(n[0], n[1])

    def bearing_and_distance(self, current_xy, goal_xy):
        dx = goal_xy[0] - current_xy[0]
        dy = goal_xy[1] - current_xy[1]
        bearing = math.atan2(dy, dx)
        dist = math.hypot(dx, dy)
        return bearing, dist

    def wrap_to_pi(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def sort_landmarks_by_distance(self, current_xy, goals):
        return sorted(
            goals,
            key=lambda g: math.hypot(
                g["position"][0] - current_xy[0], g["position"][1] - current_xy[1]
            ),
        )
