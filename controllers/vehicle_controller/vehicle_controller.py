from typing import Any
from numpy.typing import NDArray
from ekf import EKF
from lidar_wrapper import LidarWrapper
from kinematics import Kinematics
from controller import Robot
import numpy as np
from numpy import float64
import time


class PioneerDXController(Robot):
    def __init__(
        self, timestep: int, lidar_noise: float, landmarks_number: int
    ) -> None:
        super(PioneerDXController, self).__init__()
        self.timestep: int = timestep
        self.kinematics: Kinematics = Kinematics(
            wheel_radius=(0.19 / 2), wheel_distance=0.38, global_timestep=self.timestep
        )

        self.camera = self.getDevice("camera")
        self.camera.enable(self.timestep)

        self.left_wheel = self.getDevice("left wheel")
        self.right_wheel = self.getDevice("right wheel")

        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))

        self.left_wheel.setVelocity(-0.1)
        self.right_wheel.setVelocity(0.2)

        self.lidar = self.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        self.lidar.noise = lidar_noise
        self.lidar.enablePointCloud()

        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)

        self.state_vector: NDArray[float64] = np.zeros(
            (1, 3 + landmarks_number * 3)
        ).transpose()
        self.state_covariance_matrix: NDArray[float64] = (
            np.eye(self.state_vector.shape[0]) * 1e-6
        )

    def run(self) -> None:
        # Motion covariance matrix
        R_matrix = np.diag([0.001, 0.001, 0.0001])
        # Observation Covariance matrix
        Q_matrix = np.diag([0.001, 0.0001, 0.0])
        lidar_wrapper: LidarWrapper = LidarWrapper()
        ekf_wrapper: EKF = EKF(self.timestep, landmarks_number, R_matrix, Q_matrix)

        while self.step(self.timestep) != -1:
            print(f">>> GPS {self.gps.getValues()}")
            pc: bytearray = self.lidar.getPointCloud()
            cluster = lidar_wrapper.get_largest_cluster(pc)
            distance: float | None = lidar_wrapper.get_distance(cluster)
            (v_t, omega_t, theta_t) = self.kinematics.get_kinematics(
                self.left_wheel.getVelocity(),
                self.right_wheel.getVelocity(),
                self.compass.getValues(),
            )
            control_vector: NDArray[float64] = self.build_u_vector(v_t, omega_t)
            measurement_vector: NDArray[float64] = np.array([[distance, 0.1, 1.0]])
            new_state_vector: NDArray[float64] = ekf_wrapper.predict(
                self.state_vector,
                self.state_covariance_matrix,
                control_vector,
                measurement_vector,
            )
            # print(f"ts: {robot.timestep}")
            print(f"distance: {distance}")

    def build_u_vector(self, v_t: float, omega_t: float) -> NDArray[np.float64]:
        return np.array([[v_t, omega_t]])


if __name__ == "__main__":
    landmarks_number: int = 1
    timestep: int = 1000
    lidar_noise: float = 0.0
    controller: PioneerDXController = PioneerDXController(
        timestep, lidar_noise, landmarks_number
    )
    controller.run()
