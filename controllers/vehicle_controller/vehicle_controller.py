from datetime import datetime, timedelta
from typing import Any
from numpy.typing import NDArray
from apriltag_pose import AprilTagPoseEstimator
from ekf import EKF
from lidar_wrapper import LidarWrapper
from kinematics import Kinematics
from controller import Robot
import numpy as np
from numpy import float64

import helper


class PioneerDXController(Robot):
    def __init__(
        self, timestep: int, lidar_noise: float, landmarks_number: int
    ) -> None:
        super(PioneerDXController, self).__init__()
        self.timestep: int = timestep
        self.kinematics: Kinematics = Kinematics(
            wheel_radius=(0.19 / 2), wheel_distance=0.38, global_timestep=self.timestep
        )

        self.left_wheel = self.getDevice("left wheel")
        self.right_wheel = self.getDevice("right wheel")

        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))

        self.left_wheel.setVelocity(0.0)
        self.right_wheel.setVelocity(0.9)

        self.lidar = self.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        self.lidar.noise = lidar_noise
        self.lidar.enablePointCloud()

        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)

        self.camera = self.getDevice("kinect color")
        self.camera.enable(self.timestep)

        # mu [x,y,theta] (3x1)
        self.state_vector: NDArray[float64] = np.zeros((3, 1))
        # sigma (3x3)
        self.state_covariance_matrix: NDArray[float64] = np.eye(3) * 1e-6

        # landmark control
        self.REUSE_LANDMARK_THRESHOLD = 4.0
        self.landmark_sightings: dict[int, datetime] = {}

    def run(self) -> None:
        # Motion covariance matrix
        R_matrix = np.diag([0.001, 0.001, 0.0001])

        # Observation Covariance matrix [range, bearing, signature].
        # We need to initialize the signature with a very small value to prevent
        # errors while calculating the inverse
        Q_matrix = np.diag([0.001, 0.0025, 1e-6])

        lidar_wrapper: LidarWrapper = LidarWrapper()
        ekf_wrapper: EKF = EKF(self.timestep, R_matrix, Q_matrix)

        aprilTagEstimator: AprilTagPoseEstimator = AprilTagPoseEstimator(
            self.camera.getWidth(), self.camera.getHeight(), self.camera.getFov()
        )

        # Setting the correct wheels velocity to circular motion
        self.left_wheel.setVelocity(11.66)
        self.right_wheel.setVelocity(11.96)

        # ekf to world frame reference
        robot_initial_xy: list[float] | None = None
        robot_initial_theta: float | None = None

        step_count: int = 0
        while self.step(self.timestep) != -1:
            if robot_initial_xy is None and robot_initial_theta is None:
                robot_initial_xy = self.gps.getValues()[:2]
                compass_x, compass_y, _ = self.compass.getValues()
                robot_initial_theta = helper.get_heading(compass_x, compass_y)

            compass_data: list[float] = self.compass.getValues()
            v_t, omega_t, theta_t = self.kinematics.get_kinematics(
                self.left_wheel.getVelocity(),
                self.right_wheel.getVelocity(),
                compass_data,
            )

            # u vector linear and angular velocities [v_t, omega_t]
            control_vector: NDArray[float64] = self.build_u_vector(v_t, omega_t)

            # EKF prediction step
            self.state_vector, self.state_covariance_matrix = ekf_wrapper.predict(
                self.state_vector, self.state_covariance_matrix, control_vector
            )
            camera_image: bytearray = self.camera.getImage()
            tags_found: list[Any] = aprilTagEstimator.estimate_pose(camera_image)

            # We check if there are found landmarks. Otherwise, we keep moving
            if tags_found:
                # Create the z matrix based on the tags identification [distance, bearing, signature]
                z_matrix = np.empty((0, 3))
                # Create the correspondence vector c
                c_vector = np.empty(0)

                for i in range(0, len(tags_found)):
                    t = tags_found[i]
                    t_id = t["tag_id"]
                    lm_x, lm_y, _ = t["translation"]
                    lm_distance = t["distance"]

                    # Avoif using the same landmark several times while moving
                    if self.should_measure_landmark(t_id):
                        print(
                            f"! LANDMARK AT X:{lm_x:.4f}\tY:{lm_y:.4f}\t DISTANCE:{lm_distance:.4f}"
                        )

                        t_vector = np.array([t["distance"], 0.2, t_id])
                        z_matrix = np.vstack((z_matrix, t_vector))
                        c_vector = np.append(c_vector, t_id)

                        # We have to "fix" the measurent vector because each column is a different landmark
                        z_matrix = z_matrix.T

                        # EKF correction step
                        self.state_vector, self.state_covariance_matrix = (
                            ekf_wrapper.correct(
                                self.state_vector,
                                self.state_covariance_matrix,
                                z_matrix,
                                c_vector,
                            )
                        )

            if step_count % 10 == 0:
                gps_x, gps_y, _ = self.gps.getValues()
                compass_x, compass_y = compass_data[:2]
                robot_heading = helper.get_heading(compass_x, compass_y)
                print(
                    f"> REAL POSE X:{gps_x:.4f}  Y:{gps_y:.4f}  H:{robot_heading:.4f}"
                )

                ekf_x, ekf_y, ekf_heading = helper.ekf_to_global(
                    self.state_vector[:3], robot_initial_xy, robot_initial_theta
                )
                print(f"> EKF  POSE X:{ekf_x:.4f}  Y:{ekf_y:.4f}  H:{ekf_heading:.4f}")

            step_count += 1

    def build_u_vector(self, v_t: float, omega_t: float) -> NDArray[np.float64]:
        return np.array([[v_t, omega_t]])

    def should_measure_landmark(self, landmark_id: int) -> bool:
        if not landmark_id in self.landmark_sightings.keys():
            self.landmark_sightings[landmark_id] = datetime.now()
            return True
        last_seen = self.landmark_sightings[landmark_id]
        if (datetime.now() - last_seen).total_seconds() > self.REUSE_LANDMARK_THRESHOLD:
            self.landmark_sightings[landmark_id] = datetime.now()
            return True
        return False


if __name__ == "__main__":
    landmarks_number: int = 1
    timestep: int = 100
    lidar_noise: float = 0.0
    controller: PioneerDXController = PioneerDXController(
        timestep, lidar_noise, landmarks_number
    )
    controller.run()
