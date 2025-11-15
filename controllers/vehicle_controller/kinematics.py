#!/usr/bin/env python3
import numpy as np
import math


class Kinematics:
    def __init__(
        self, wheel_radius: float, wheel_distance: float, global_timestep: int
    ):
        self.wr: float = wheel_radius
        self.wd: float = wheel_distance
        self.ts: float = global_timestep  # / 1000.0

    def get_kinematics(
        self,
        l_velocity: float | None,
        r_velocity: float | None,
        compass: list[float] | None,
    ) -> tuple[float, float, float]:
        if l_velocity is None or r_velocity is None or compass is None:
            raise Exception("Left and right velocity must be provided")
        # print(f"Left wheel velocity: {l_velocity} / Right wheel velocity: {r_velocity} / Compass: {compass}")
        v_t: float = (l_velocity + r_velocity) / 2.0
        omega_t: float = (r_velocity - l_velocity) / self.wd
        theta_t: float = math.atan2(compass[0], compass[1])

        print(f"Velocity: {v_t} / Angular velocity: {omega_t} / Heading: {theta_t}")
        return (v_t, omega_t, theta_t)
