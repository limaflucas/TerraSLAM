from typing import Any
from numpy.typing import NDArray
from apriltag_pose import AprilTagPoseEstimator
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

        # self.camera = self.getDevice("camera")
        # self.camera.enable(self.timestep)

        self.left_wheel = self.getDevice("left wheel")
        self.right_wheel = self.getDevice("right wheel")

        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))

        self.left_wheel.setVelocity(0.57)
        self.right_wheel.setVelocity(0.60)

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

        ### Autonomous navigation
        self.MAX_SPEED: float = 1.2
        self.states: list[str] = ["forward", "turn_right", "turn_left"]
        self.current_state: str = self.states[0]
        self.counter: int = 0
        self.COUNTER_MAX: int = 3
        self.puck_ground_sensors: list[Any] = []
        for i in ["gs0", "gs2"]:
            sensor = self.getDevice(i)
            sensor.enable(self.timestep)
            self.puck_ground_sensors.append(sensor)

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

        while self.step(self.timestep) != -1:
            self.navigate_robot()
            print(f">>> GPS {self.gps.getValues()}")
            print(f">>> INTERNAL STATE: {self.state_vector[:,0]}")

            v_t, omega_t, theta_t = self.kinematics.get_kinematics(
                self.left_wheel.getVelocity(),
                self.right_wheel.getVelocity(),
                self.compass.getValues(),
            )

            print(f">>> COMPASS {theta_t}")
            # u vector linear and angular velocities [v_t, omega_t]
            control_vector: NDArray[float64] = self.build_u_vector(v_t, omega_t)

            # EKF prediction step
            self.state_vector, self.state_covariance_matrix = ekf_wrapper.predict(
                self.state_vector, self.state_covariance_matrix, control_vector
            )
            camera_image: bytearray = self.camera.getImage()
            tags_found: list[Any] = aprilTagEstimator.estimate_pose(camera_image)

            # We check if there are found landmarks. Otherwise, we keep moving
            if len(tags_found) < 1:
                print("No landmarks detected in this step")
                continue

            # Create the z matrix based on the tags identification [distance, bearing, signature]
            z_matrix = np.empty((0, 3))
            # Create the correspondence vector c
            c_vector = np.empty(0)
            for i in range(0, len(tags_found)):
                t = tags_found[i]
                print(f"> LANDMARK AT {t['translation']} - DISTANCE {t['distance']}")
                t_id = t["tag_id"]
                t_vector = np.array([t["distance"], 0.2, t_id])
                z_matrix = np.vstack((z_matrix, t_vector))
                c_vector = np.append(c_vector, t_id)

            # We have to "fix" the measurent vector because each column is a different landmark
            z_matrix = z_matrix.T

            # EKF correction step
            self.state_vector, self.state_covariance_matrix = ekf_wrapper.correct(
                self.state_vector, self.state_covariance_matrix, z_matrix, c_vector
            )

            # measurement_vector: NDArray[float64] = np.array([[distance, 0.1, 1.0]])
            # correspondence_vector: NDArray[float64] = np.array([[1]])

    def build_u_vector(self, v_t: float, omega_t: float) -> NDArray[np.float64]:
        return np.array([[v_t, omega_t]])

    def navigate_robot(self) -> None:
        line_right = self.puck_ground_sensors[1].getValue() > 600
        line_left = self.puck_ground_sensors[0].getValue() > 600

        if self.current_state == "forward":
            # Action for the current state: update speed variables
            leftSpeed = self.MAX_SPEED
            rightSpeed = self.MAX_SPEED

            # check if it is necessary to update current_state
            if line_right and not line_left:
                self.current_state = "turn_right"
                self.counter = 0
            elif line_left and not line_right:
                self.current_state = "turn_left"
                self.counter = 0

        if self.current_state == "turn_right":
            # Action for the current state: update speed variables
            leftSpeed = 0.8 * self.MAX_SPEED
            rightSpeed = 0.4 * self.MAX_SPEED

            # check if it is necessary to update current_state
            if self.counter == self.COUNTER_MAX:
                self.current_state = "forward"

        if self.current_state == "turn_left":
            # Action for the current state: update speed variables
            leftSpeed = 0.4 * self.MAX_SPEED
            rightSpeed = 0.8 * self.MAX_SPEED

            # check if it is necessary to update current_state
            if self.counter == self.COUNTER_MAX:
                self.current_state = "forward"

        self.counter += 1
        self.left_wheel.setVelocity(leftSpeed)
        self.right_wheel.setVelocity(rightSpeed)


if __name__ == "__main__":
    landmarks_number: int = 1
    timestep: int = 100
    lidar_noise: float = 0.0
    controller: PioneerDXController = PioneerDXController(
        timestep, lidar_noise, landmarks_number
    )
    controller.run()
