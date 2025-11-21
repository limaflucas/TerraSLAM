from numpy import float64
import numpy as np
import math

from numpy.typing import NDArray

"""
The code implemented in this file was created following the pseudo-code in the book Probabilistic Robotics (by Sebastian Thrun, Wolfram Burgard, Dieter Fox)
It can be found at page 314 - Table 10.1
"""


class EKF:
    def __init__(
        self,
        timestep: int,
        motion_covariance: NDArray[float64],
        observation_covariance: NDArray[float64],
    ) -> None:
        self._Rt: NDArray[float64] = motion_covariance
        self._Qt: NDArray[float64] = observation_covariance
        self.timestep: int = timestep
        self.landmarks_map: dict[int, int] = {}

    def predict(
        self,
        state_vector: NDArray[float64],
        state_covariance_matrix: NDArray[float64],
        control_vector: NDArray[float64],
    ) -> tuple[NDArray[float64], NDArray[float64]]:

        # print( f"Begining prediction with: MU\n{state_vector}\n Sigma:\n{state_covariance_matrix}\n U: \n{control_vector}")

        v_t: float64 = control_vector[0][0]
        omega_t: float64 = control_vector[0][1]
        theta_0: float64 = state_vector[2][0]

        v_t_div_omega_t: float64 = v_t / omega_t
        theta_t_omega_t: float64 = theta_0 + (omega_t * self.timestep)

        x: float64 = (
            -v_t_div_omega_t * math.sin(theta_0)
        ) + v_t_div_omega_t * math.sin(theta_t_omega_t)
        y: float64 = (v_t_div_omega_t * math.cos(theta_0)) - v_t_div_omega_t * math.cos(
            theta_t_omega_t
        )
        theta: float64 = omega_t * self.timestep

        # G_t Jacobian Motion Model
        g_x = -v_t_div_omega_t * math.cos(theta_0) + v_t_div_omega_t * math.cos(
            theta_t_omega_t
        )
        g_y = -v_t_div_omega_t * math.sin(theta_0) + v_t_div_omega_t * math.sin(
            theta_t_omega_t
        )

        Fx = np.eye(3, state_vector.shape[0])

        # mu bar math
        motion: NDArray[float64] = np.array([[x, y, theta]]).T
        new_state_vector: NDArray[float64] = state_vector + np.matmul(Fx.T, motion)
        # we have to normalized, otherwise the heading is wild
        new_state_vector[2, 0] = self._normalize_heading(new_state_vector[2, 0])
        # print(f"MU_bar: \n{new_state_vector}")
        #
        g_t = np.zeros((3, 3))
        g_t[:, 2] = np.array([g_x, g_y, 0.0]).T
        size = state_covariance_matrix.shape[0]
        I = np.eye(size)
        G_t: NDArray[float64] = I + np.matmul(np.matmul(Fx.T, g_t), Fx)

        # print(f"Jacobian motion model:\n{G_t}")

        # Sigma bar
        Sigma_t: NDArray[float64] = np.matmul(
            G_t, np.matmul(state_covariance_matrix, G_t.T)
        ) + np.matmul(np.matmul(Fx.T, self._Rt), Fx)

        # print(f">>> PREDICTION MU BAR:\n {new_state_vector}")

        return (new_state_vector, Sigma_t)

    def correct(
        self,
        state_matrix: NDArray[float64],
        state_covariance_matrix: NDArray[float64],
        measurement_vector: NDArray[float64],
        correspondence_vector: NDArray[float64],
    ) -> tuple[NDArray[float64], NDArray[float64]]:

        mu: NDArray[float64] = state_matrix
        Sigma: NDArray[float64] = state_covariance_matrix

        robot_x, robot_y, robot_theta = mu[:3, 0]

        for i in range(0, measurement_vector.shape[1]):
            j: int = correspondence_vector[i]
            # print(f"Correcting: {measurement_vector[:,i]} against landmark: {j}")
            if j not in self.landmarks_map.keys():
                measurement: NDArray[float64] = measurement_vector[:, i]
                lm_distance: float = measurement[0]
                lm_bearing: float = measurement[1]
                lm_x: float = robot_x + lm_distance * math.cos(lm_bearing + robot_theta)
                lm_y: float = robot_y + lm_distance * math.sin(lm_bearing + robot_theta)
                lm_s: int = measurement[2]
                # print(f"DEBUGG measurement >>> {robot_x, robot_y}")
                print(f"Inserting new landmark X:{lm_x:.4f} Y:{lm_y:.4f} S:{lm_s}")

                # Augment the state matrix to hold the new landmark
                mu = np.vstack((mu, np.array([lm_x, lm_y, lm_s]).reshape(3, 1)))
                # print(f"Augmented state matrix with landmark {j}: {state_matrix}")
                self.landmarks_map[j] = len(self.landmarks_map.keys()) + 1

                # Augment the state matrix covariance to hold the new landmark covariance
                size: int = Sigma.shape[0]
                augmented_sigma = np.eye(size + 3) * 1e-6
                # Tricky part. We need to copy the old values to the new matrix
                augmented_sigma[0:size, 0:size] = Sigma
                # Because this is a new landmark we have a large covariance associated (just following the book)
                augmented_sigma[size, size] = 1e6
                augmented_sigma[size + 1, size + 1] = 1e6
                Sigma = augmented_sigma

            # delta distance and q math
            lm_start_index, lm_end_index = self._get_landmark_index(j)
            lm_xy: NDArray[float64] = mu[lm_start_index:lm_end_index, 0]
            robot_xy: NDArray[float64] = np.array([robot_x, robot_y])
            delta: NDArray[float64] = lm_xy - robot_xy
            q: float = np.matmul(delta.T, delta)

            # z hat and innovation math
            delta_x, delta_y = delta
            z_hat_values = [math.sqrt(q), math.atan2(delta_y, delta_x) - robot_theta, j]
            z_hat = np.array(z_hat_values).reshape(3, 1)
            measurement = measurement_vector[:, i : i + 1]
            innovation = measurement - z_hat
            innovation[1] = self._normalize_heading(innovation[1])

            # Measurement Jacobian Fxj and Ht
            F_xj = np.zeros((6, mu.shape[0]))
            # Identity to the robots pose
            F_xj[0:3, 0:3] = np.eye(3)
            # Identity to the landmark. We need to +3 because of x, y, signature
            F_xj[3:6, lm_start_index : lm_start_index + 3] = np.eye(3)
            H_inner = self._get_H_inner_matrix(q, delta_x, delta_y)
            H = (1 / q) * np.matmul(H_inner, F_xj)

            # Kalman gain math K
            # Part 1 - Sigma_bar * H.T
            k_pt1 = np.matmul(Sigma, H.T)

            # Part 2 - (H * Sigma_bar * H.T + Q)^-1 > we need to break into smaller blocks
            ### Part 2_1 - H * Sigma_bar
            k_pt2_1 = np.matmul(H, Sigma)

            ### Part 2_2 - (H * Sigma_bar * H.T)
            k_pt2_2 = np.matmul(k_pt2_1, H.T)

            ### Part 2_3 - (H * Sigma_bar * H.T + Qt)
            k_pt2_3 = k_pt2_2 + self._Qt

            ### Part 2_4 - (H * Sigma_bar * H.T + Qt)^-1 (inverse)
            k_pt2 = np.linalg.inv(k_pt2_3)

            ### Part 3 - Part 1 * Part 2
            K = np.matmul(k_pt1, k_pt2)

            # State update and heading normalization
            mu_bar: NDArray[float64] = mu + np.matmul(K, innovation)
            mu_bar[2] = self._normalize_heading(mu_bar[2])

            # State covariance update
            I = np.eye(Sigma.shape[0])
            pre_Sigma: NDArray[float64] = I - np.matmul(K, H)
            Sigma_bar: NDArray[float64] = np.matmul(pre_Sigma, Sigma)

            # Slightly different from the book
            mu = mu_bar
            Sigma = Sigma_bar

        # print(f">>> CORRECTION MU BAR:\n{new_mu}")
        return (mu, Sigma)

    def _normalize_heading(self, angle: float) -> float:
        """Normalizes the angle to the [-pi, pi] range"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _get_landmark_index(self, lm_index: int) -> tuple[int, int]:
        index = self.landmarks_map[lm_index]
        state_vector_start_index = index * 3
        state_vector_end_index = state_vector_start_index + 2

        return (state_vector_start_index, state_vector_end_index)

    def _get_H_inner_matrix(
        self, q: float, delta_x: float, delta_y: float
    ) -> NDArray[float64]:
        sqrt_q = np.sqrt(q)
        H_inner = np.zeros((3, 6))

        # Row 1
        H_inner[0, 0] = -sqrt_q * delta_x
        H_inner[0, 1] = -sqrt_q * delta_y
        H_inner[0, 2] = 0
        H_inner[0, 3] = sqrt_q * delta_x
        H_inner[0, 4] = sqrt_q * delta_y
        H_inner[0, 5] = 0

        # Row 2
        H_inner[1, 0] = delta_y
        H_inner[1, 1] = -delta_x
        H_inner[1, 2] = -q
        H_inner[1, 3] = -delta_y
        H_inner[1, 4] = delta_x
        H_inner[1, 5] = 0

        # Row 3
        H_inner[2, 0] = 0
        H_inner[2, 1] = 0
        H_inner[2, 2] = 0
        H_inner[2, 3] = 0
        H_inner[2, 4] = 0
        H_inner[2, 5] = q

        return H_inner
