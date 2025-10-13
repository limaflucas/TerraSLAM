#!/usr/bin/env python3
import numpy as np
import math


class EKF:
    def __init__(self, kinematics=None):
        self._chisq_threshold = 5.991  # 95% confidence for 2 DoF
        # Kinematics state and configuration
        self.kinematics = kinematics
        # x-state matrix: [x_r, y_r, theta_r, x_m1, y_m1, x_m2, y_m2, ...]
        self.x_state = np.array(self.kinematics.get_pose())
        # P-matrix - covariance matrix used for KF uncertainty
        # self.state = np.array([[], [0.0], [0.0]])
        self.covariance = np.diag([1.0, 1.0, 1.0]) * 1e-6
        # Q-matrix -  Motion noise covariance
        self.motion_noise = np.diag([0.001, 0.001, 0.0001])
        # Q-matrix - Sensor noise covariance - Tune for Lidar/Camera range
        self.sensor_noise = np.diag(
            [0.01, 0.0025]
        )  # Variance of range (m^2) and bearing (rad^2)
        self.landmark_number = 0
        self.map_descriptors = []

    def predict(self, l_velocity=None, r_velocity=None):
        delta_d, delta_theta = self.kinematics.get_linear_distance(
            l_velocity, r_velocity
        )

        theta = self.x_state[2, 0]
        # State transition matrix for robot (1 row, 3 cels) and the landmarks coordinates (x, y)
        f_matrix = np.identity(3 + 2 * self.landmark_number)
        # Initializes with the new robot (x, y) estimate
        f_matrix[0, 2] = -delta_d * math.sin(theta + delta_theta / 2.0)
        f_matrix[1, 2] = delta_d * math.cos(theta + delta_theta / 2.0)

        # Calculates new x, y, theta and adjust the bearing
        new_x = self.x_state[0, 0] + (delta_d * math.cos(theta + delta_theta / 2.0))
        new_y = self.x_state[1, 0] + (delta_d * math.sin(theta + delta_theta / 2.0))
        new_theta = theta + delta_theta
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
        self.kinematics.update_pose(new_x, new_y, new_theta)
        self.x_state[:3] = self.kinematics.get_pose()

        # print(f"---> pose {self.kinematics.get_pose()}")

        Q_motion = np.zeros_like(self.covariance)
        Q_motion[:3, :3] = self.motion_noise
        self.covariance = f_matrix @ self.covariance @ f_matrix.T + Q_motion

    def h(self, landmark_num):
        x_r, y_r, theta_r = self.x_state[:3, 0]
        m_index = 3 + 2 * landmark_num
        # x_m, y_m = self.x_state[m_index, m_index + 1]
        x_m = self.x_state[m_index, 0]
        y_m = self.x_state[m_index + 1, 0]

        r, phi = self._get_r_and_phi(x_r, y_r, theta_r, x_m, y_m)
        return np.array([[r], [phi]])

    def H_k(self, landmark_num):
        x_r, y_r, theta_r = self.x_state[:3, 0]
        m_index = 3 + 2 * landmark_num
        # x_m, y_m = self.x_state[m_index, m_index + 1]
        x_m = self.x_state[m_index, 0]
        y_m = self.x_state[m_index + 1, 0]
        dx, dy = self._get_distances(x_r, y_r, x_m, y_m)
        r, _ = self._get_r_and_phi(x_r, y_r, theta_r, x_m, y_m)
        r_sq = r * r

        # J_r (2x3) and J_m (2x2) derivation based on h = [r; phi]
        J_r = np.array([[-dx / r, -dy / r, 0.0], [dy / r_sq, -dx / r_sq, -1.0]])
        J_m = np.array([[dx / r, dy / r], [-dy / r_sq, dx / r_sq]])

        # Assemble the full H_k matrix (2 x (3 + 2N))
        H_k = np.zeros((2, 3 + 2 * self.landmark_number))
        H_k[:, 0:3] = J_r  # Robot block
        H_k[:, m_index : m_index + 2] = J_m  # Landmark j block

        return H_k

    def correct(self, z_k, landmark_index):
        # Implementation of the EKF Correction Step (Observation Model)
        H = self.H_k(landmark_index)
        S = H @ self.covariance @ H.T + self.sensor_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)

        residual = z_k - self.h(landmark_index)
        residual[1, 0] = math.atan2(math.sin(residual[1, 0]), math.cos(residual[1, 0]))

        # 2. Update State Vector (x_k = x_bar + K_k * residual)
        self.x_state = self.x_state + K @ residual

        # Update robot pose copy
        self.kinematics.update_pose(
            self.x_state[0, 0], self.x_state[1, 0], self.x_state[2, 0]
        )

        # 3. Update Covariance Matrix (P_k = (I - K_k * H_k) * P_bar)
        I = np.identity(3 + 2 * self.landmark_number)
        self.covariance = (I - K @ H) @ self.covariance

    def augment_state(self, z_k, descriptor):
        # Implementation of the Landmark Initialization Step
        x_r, y_r, theta_r = self.x_state[:3, 0]
        r, phi = z_k[0, 0], z_k[1, 0]

        # 1. Inverse Observation Model: Calculate global position of new landmark
        x_m = x_r + r * math.cos(theta_r + phi)
        y_m = y_r + r * math.sin(theta_r + phi)

        # 2. Augment State Vector (x)
        new_landmark_state = np.array([[x_m], [y_m]])
        self.x_state = np.vstack([self.x_state, new_landmark_state])

        # 3. Augment Covariance Matrix (P)
        # Jacobians for the Inverse Observation Model
        J_r = np.array(
            [
                [1.0, 0.0, -r * math.sin(theta_r + phi)],
                [0.0, 1.0, r * math.cos(theta_r + phi)],
            ]
        )
        J_z = np.array(
            [
                [math.cos(theta_r + phi), -r * math.sin(theta_r + phi)],
                [math.sin(theta_r + phi), r * math.cos(theta_r + phi)],
            ]
        )

        # New landmark self-covariance P_m,m
        P_r_r = self.covariance[0:3, 0:3]
        P_m_m = J_r @ P_r_r @ J_r.T + J_z @ self.sensor_noise @ J_z.T

        # New landmark-robot cross-covariance P_m,r
        P_m_r = J_r @ P_r_r

        # Construct the augmented P_covariance matrix
        old_size = 3 + 2 * self.landmark_number
        new_size = old_size + 2
        P_new = np.zeros((new_size, new_size))

        P_new[:old_size, :old_size] = self.covariance
        P_new[old_size:new_size, old_size:new_size] = P_m_m
        P_new[old_size:new_size, 0:3] = P_m_r
        P_new[0:3, old_size:new_size] = P_m_r.T

        # P_m,m_old and P_m_old,m blocks (correlations with existing landmarks)
        P_new[old_size:new_size, 3:old_size] = P_m_r @ self.covariance[0:3, 3:old_size]
        P_new[3:old_size, old_size:new_size] = P_new[old_size:new_size, 3:old_size].T

        self.covariance = P_new

        # 4. Update counts and storage
        # Ensure descriptor is 1D for storage key
        self.map_descriptors.append(descriptor.flatten())
        self.landmark_number += 1

    def find_data_association(self, current_descriptor, z_k):
        if self.landmark_number == 0:
            return -1

        min_mahalanobis_dist_sq = float("inf")
        best_match_index = -1

        for j in range(self.landmark_number):
            known_descriptor = self.map_descriptors[j]
            hamming_dist = np.sum(current_descriptor != known_descriptor)
            if hamming_dist > 50:  # Tune this threshold based on visual robustness
                continue
            H = self.H_k(j)
            S = H @ self.covariance @ H.T + self.sensor_noise
            residual = z_k - self.h(j)
            residual[1, 0] = math.atan2(
                math.sin(residual[1, 0]), math.cos(residual[1, 0])
            )
            mahal_dist_sq = residual.T @ np.linalg.inv(S) @ residual
            if mahal_dist_sq < min_mahalanobis_dist_sq:
                min_mahalanobis_dist_sq = mahal_dist_sq
                best_match_index = j

        if best_match_index != -1 and min_mahalanobis_dist_sq < self._chisq_threshold:
            return best_match_index
        else:
            return -1

    def _get_r_and_phi(self, x_r, y_r, theta_r, x_m, y_m):
        dx, dy = self._get_distances(x_r, y_r, x_m, y_m)
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) - theta_r
        # Normalize bearing
        phi = math.atan2(math.sin(phi), math.cos(phi))

        return r, phi

    def _get_distances(self, x_r, y_r, x_m, y_m):
        dx, dy = x_m - x_r, y_m - y_r
        return dx, dy
