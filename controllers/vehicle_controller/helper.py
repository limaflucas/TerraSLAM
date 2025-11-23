import array
import math

from numpy import float64
from numpy.typing import NDArray
import numpy as np


def local_to_global(robot_pose: NDArray[float64], local_point: NDArray[float64]):
    rx, ry, r_theta = robot_pose
    lx, ly = local_point

    global_x = rx + lx * math.cos(r_theta) - ly * math.sin(r_theta)
    global_y = ry + lx * math.sin(r_theta) + ly * math.cos(r_theta)

    return (global_x, global_y)


def ekf_to_global(
    ekf_pose: NDArray[float64], initial_xy: list[float], initial_theta: float
):
    x_ekf, y_ekf, theta_ekf = ekf_pose.flatten()
    x_init, y_init = initial_xy
    theta_init = initial_theta

    # print(f">>> Parameters: x {x_ekf} y {y_ekf} theta {theta_ekf}")

    # 1. Rotate the EKF translation to align with Global frame
    # (The EKF X-axis is aligned with the robot's initial heading)
    x_rot = x_ekf * math.cos(theta_init) - y_ekf * math.sin(theta_init)
    y_rot = x_ekf * math.sin(theta_init) + y_ekf * math.cos(theta_init)

    # 2. Translate by the initial position
    x_global = x_init + x_rot
    y_global = y_init + y_rot

    # 3. Update heading
    theta_global = theta_init + theta_ekf
    theta_global = math.atan2(math.sin(theta_global), math.cos(theta_global))

    return np.array([x_global, y_global, theta_global])


def get_heading(x: float, y: float) -> float:
    return math.atan2(x, y)
