#!/usr/bin/env python3
import numpy as np


class Kinematics:
    def __init__(self, x, y, theta, wheel_radius, wheel_distance, global_timestep):
        self.x = x
        self.y = y
        self.theta = theta
        self.wr = wheel_radius
        self.wd = wheel_distance
        self.ts = global_timestep / 1000.0

    def get_linear_distance(self, l_velocity=None, r_velocity=None):
        if l_velocity is None or r_velocity is None:
            raise Exception("Left and right velocity must be provided")
        delta_l = l_velocity * self.wr * self.ts
        delta_r = r_velocity * self.wr * self.ts

        delta_d = (delta_l + delta_r) / 2.0
        delta_theta = (delta_l - delta_r) / self.wd

        return delta_d, delta_theta

    def get_pose(self):
        return np.array([[self.x], [self.y], [self.theta]])

    def update_pose(self, x=None, y=None, theta=None):
        if x is None or y is None or theta is None:
            raise Exception("x, y, theta must be provided")
        self.x = x
        self.y = y
        self.theta = theta
