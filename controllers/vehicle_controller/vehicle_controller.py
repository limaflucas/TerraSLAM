from datetime import datetime
from typing import Any
from plot_wrapper import SLAMGraph
from numpy.typing import NDArray
from apriltag_pose import AprilTagPoseEstimator
from ekf import EKF
from kinematics import Kinematics
from controller import Robot
import numpy as np
from numpy import float64
import matplotlib.pyplot as plt
import helper


class PioneerDXController(Robot):
    def __init__(
        self, timestep: int, lidar_noise: float, landmarks_number: int
    ) -> None:
        super(PioneerDXController, self).__init__()
        self.timestep: int = timestep

        self.left_wheel = self.getDevice("left wheel")
        self.right_wheel = self.getDevice("right wheel")

        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))

        self.left_wheel.setVelocity(0.0)
        self.right_wheel.setVelocity(0.0)

        self.compass = self.getDevice("compass")
        self.compass.enable(self.timestep)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)

        self.camera = self.getDevice("kinect color")
        self.camera.enable(self.timestep)

        ###
        #  Kinematics Model Initialization
        ###
        self.kinematics: Kinematics = Kinematics(
            wheel_radius=0.0976, wheel_distance=0.33
        )

        ###
        #  EKF Initialization
        ###
        # State Vector (MU) [x,y,theta] (3x1)
        self.state_vector: NDArray[float64] = np.zeros((3, 1))
        # State Covariance Matrix (SIGMA) (3x3)
        self.state_covariance_matrix: NDArray[float64] = np.eye(3) * 1e-6
        # Motion Covariance Matrix (R)
        R_matrix = np.diag([0.001, 0.0025, 0.0001])
        # Observation Covariance Matrix (Q) [range, bearing, signature].
        # We need to initialize the signature with a very small value to prevent errors while calculating the inverse
        Q_matrix = np.diag([0.001, 0.0025, 1e-6])
        self.ekf: EKF = EKF(self.timestep / 1000.0, R_matrix, Q_matrix)

        # landmark control
        self.REUSE_LANDMARK_THRESHOLD = 2.0
        self.landmark_sightings: dict[int, datetime] = {}

        ###
        #  AprilTags INITIALIZATION
        ###
        self.aprilTagEstimator: AprilTagPoseEstimator = AprilTagPoseEstimator(
            self.camera.getWidth(), self.camera.getHeight(), self.camera.getFov()
        )

        # Graph Plotting
        self.slam_graph: SLAMGraph = SLAMGraph(on_keyboard_press)

    def run(self) -> None:

        print("!!! Press 'p' on the keyboard to interact with the plots !!!")

        # ekf to world frame reference
        robot_initial_xy: list[float] | None = None
        robot_initial_theta: float | None = None

        step_count: int = 0
        while self.step(self.timestep) != -1:

            ###
            # City world - Setting the correct wheels velocity to circular motion
            ###
            # self.left_wheel.setVelocity(11.66)
            # self.right_wheel.setVelocity(11.96)

            self.testing_motion(step_count)

            # Plot interaction only
            while simulation_paused:
                plt.pause(0.1)

            gps_x, gps_y, gps_z = self.gps.getValues()
            compass_x, compass_y, compass_z = self.compass.getValues()
            compass_theta = np.atan2(compass_x, compass_z)
            robot_x, robot_y, robot_theta = self.state_vector.flatten()[:3]

            print(f"> GPS X:{gps_x:.4f}  Y:{gps_y:.4f}")
            print(f"> STATE VECTOR X:{robot_x:.4f}  Y:{robot_y:.4f}")

            # EKF graph data
            ekf_predict_x, ekf_predict_y, ekf_predict_theta = None, None, None
            ekf_correct_x, ekf_correct_y, ekf_correct_theta = None, None, None

            if robot_initial_xy is None and robot_initial_theta is None:
                robot_initial_xy = [gps_x, gps_y]
                robot_initial_theta = compass_theta

            v_t, omega_t = self.kinematics.get_kinematics(
                self.left_wheel.getVelocity(), self.right_wheel.getVelocity()
            )

            print(f"> ROBOT KINEMATICS:{v_t}  Y:{omega_t}")

            # Control Vector (u) linear and angular velocities [v_t, omega_t]
            control_vector: NDArray[float64] = np.array([[v_t, omega_t]])

            # EKF prediction step
            self.state_vector, self.state_covariance_matrix = self.ekf.predict(
                self.state_vector,
                self.state_covariance_matrix,
                control_vector,
            )
            ekf_predict_x, ekf_predict_y, ekf_predict_theta = (
                self.state_vector.flatten()[:3]
            )

            print(f"> EKF PREDICT X:{ekf_predict_x:.4f}  Y:{ekf_predict_y:.4f}")

            camera_image: bytearray = self.camera.getImage()
            tags_found: list[Any] = self.aprilTagEstimator.estimate_pose(camera_image)

            # We check if there are found landmarks. Otherwise, we keep moving
            for t in tags_found:
                t_id = t["tag_id"]
                lm_x, lm_y, lm_z = t["translation"]
                lm_distance = t["distance"]

                # Avoid using the same landmark several times while moving
                if self.should_measure_landmark(t_id):
                    r_x, r_y = self.state_vector[:2]
                    print(
                        f"! LANDMARK AT X:{lm_x:.4f}\tY:{lm_y:.4f}\t DISTANCE:{lm_distance:.4f}"
                    )

                    # bearing = np.atan2(lm_y, lm_x)
                    bearing = self.kinematics.get_bearing(
                        ekf_predict_x, ekf_predict_y, ekf_predict_theta, lm_x, lm_y
                    )
                    # Create the z matrix based on the tags identification [distance, bearing, signature]
                    # We have to "fix" the measurent vector because each column is a different landmark
                    z_matrix = np.array([lm_distance, bearing, t_id]).reshape(1, 3).T
                    # Create the correspondence vector c
                    c_vector = np.array([t_id])

                    # EKF correction step
                    self.state_vector, self.state_covariance_matrix = self.ekf.correct(
                        self.state_vector,
                        self.state_covariance_matrix,
                        z_matrix,
                        c_vector,
                    )

                    ekf_correct_x, ekf_correct_y, ekf_correct_theta = (
                        self.state_vector.flatten()[:3]
                    )
                    self.collect_correction_data(
                        ekf_correct_x,
                        ekf_correct_y,
                        ekf_correct_theta,
                        robot_initial_xy,
                        robot_initial_theta,
                    )
                    print(f"> EKF CORRECT X:{ekf_correct_x:.4f}  Y:{ekf_correct_y:.4f}")

            # Data collection every 500ms
            if step_count % 5 == 0:
                adj_ekf_predict_x, adj_ekf_predict_y, adj_ekf_predict_heading = (
                    helper.ekf_to_global(
                        [ekf_predict_x, ekf_predict_y, ekf_predict_theta],
                        robot_initial_xy,
                        robot_initial_theta,
                    )
                )

                self.slam_graph.append_data_ground_truth(gps_x, gps_y)
                self.slam_graph.append_data_prediction(
                    adj_ekf_predict_x, adj_ekf_predict_y
                )

                error_x = ekf_correct_x if ekf_correct_x is not None else ekf_predict_x
                error_y = ekf_correct_y if ekf_correct_y is not None else ekf_predict_y
                error_theta = (
                    ekf_correct_theta
                    if ekf_correct_theta is not None
                    else ekf_predict_theta
                )

                adj_error_x, adj_error_y, adj_error_theta = helper.ekf_to_global(
                    [error_x, error_y, error_theta],
                    robot_initial_xy,
                    robot_initial_theta,
                )

                self.slam_graph.append_data_time((self.timestep / 1000.0) * step_count)

                self.slam_graph.append_data_heading(
                    compass_x, compass_y, adj_error_theta
                )

                self.slam_graph.append_data_pose(gps_x, gps_y, adj_error_x, adj_error_y)

            # Data plotting every 1s
            if step_count % 10 == 0:
                self.slam_graph.draw()
                plt.pause(0.001)

            step_count += 1

    def testing_motion(self, step_count) -> None:

        ## Track 01
        if int((self.getTime() / 36) % 2) == 0:
            self.left_wheel.setVelocity(0)
            self.right_wheel.setVelocity(0)
            self.left_wheel.setVelocity(2.5)
            self.right_wheel.setVelocity(2.485)
        else:
            self.left_wheel.setVelocity(0)
            self.right_wheel.setVelocity(0)
            self.left_wheel.setVelocity(-2.5)
            self.right_wheel.setVelocity(-2.485)

        ## Track 02
        # if int((self.getTime() / 34) % 2) == 0:
        #     self.left_wheel.setVelocity(0)
        #     self.right_wheel.setVelocity(0)
        #     self.left_wheel.setVelocity(2.5)
        #     self.right_wheel.setVelocity(2.485)
        # else:
        #     self.left_wheel.setVelocity(0)
        #     self.right_wheel.setVelocity(0)
        #     self.left_wheel.setVelocity(-2.5)
        #     self.right_wheel.setVelocity(-2.485)

        ## Track 03
        # if int((self.getTime() / 33) % 2) == 0:
        #     self.left_wheel.setVelocity(0)
        #     self.right_wheel.setVelocity(0)
        #     self.left_wheel.setVelocity(2.485)
        #     self.right_wheel.setVelocity(2.5)
        # else:
        #     self.left_wheel.setVelocity(0)
        #     self.right_wheel.setVelocity(0)
        #     self.left_wheel.setVelocity(-2.485)
        #     self.right_wheel.setVelocity(-2.5)

    def should_measure_landmark(self, landmark_id: int) -> bool:
        if not landmark_id in self.landmark_sightings.keys():
            self.landmark_sightings[landmark_id] = datetime.now()
            return True
        last_seen = self.landmark_sightings[landmark_id]
        if (datetime.now() - last_seen).total_seconds() > self.REUSE_LANDMARK_THRESHOLD:
            self.landmark_sightings[landmark_id] = datetime.now()
            return True
        return False

    def collect_correction_data(self, x, y, theta, initial_xy, initial_theta) -> None:
        adj_ekf_correct_x, adj_ekf_correct_y, adj_ekf_correct_heading = (
            helper.ekf_to_global(
                [x, y, theta],
                initial_xy,
                initial_theta,
            )
        )
        self.slam_graph.append_data_ekf_correction(adj_ekf_correct_x, adj_ekf_correct_y)


def on_keyboard_press(event):
    global simulation_paused
    if event.key == "p":
        simulation_paused = not simulation_paused
        print(f"!!! Simulation Paused: {simulation_paused} !!!")


if __name__ == "__main__":
    simulation_paused = False
    landmarks_number: int = 1
    timestep: int = 100
    lidar_noise: float = 0.0
    controller: PioneerDXController = PioneerDXController(
        timestep, lidar_noise, landmarks_number
    )
    controller.run()
