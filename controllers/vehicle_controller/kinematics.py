import math


class Kinematics:
    def __init__(self, wheel_radius: float, wheel_distance: float):
        self.wr: float = wheel_radius
        self.wd: float = wheel_distance

    def get_kinematics(
        self, l_velocity: float | None, r_velocity: float | None
    ) -> tuple[float, float]:
        if l_velocity is None or r_velocity is None:
            raise Exception("Left and right velocity must be provided")
        # print(f"Left wheel velocity: {l_velocity} / Right wheel velocity: {r_velocity} / Compass: {compass}")

        # wheels velocities are in rad/s. We need to convert to m/s
        v_l: float = l_velocity * self.wr
        v_r: float = r_velocity * self.wr
        v_t: float = (v_l + v_r) / 2.0
        omega_t: float = (v_r - v_l) / self.wd

        # print(f"Velocity: {v_t} / Angular velocity: {omega_t} / Heading: {theta_t}")
        return v_t, omega_t

    def get_bearing(
        self,
        robot_x: float,
        robot_y: float,
        robot_heading: float,
        object_x: float,
        object_y: float,
    ):
        alpha = math.atan2(object_y - robot_y, object_x - robot_x)
        return alpha - robot_heading
