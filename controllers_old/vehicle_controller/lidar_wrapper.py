#!/usr/bin/env python3
import numpy as np


class LidarWrapper:
    def __init__(self, min_points=3, max_points=20, max_cluster_radius=0.2):
        self.min_points = min_points
        self.max_points = max_points
        self.max_cluster_radius = max_cluster_radius

    def extract_landmarks(self, point_cloud):
        """
        Extracts landmark features from a Lidar point cloud.
        A simple clustering algorithm to group points, then filters them.
        - point_cloud: list of LidarPoint objects from lidar.getPointCloud()
        - Returns: A list of landmarks, where each landmark is a 2x1 numpy array [range; bearing]
        """
        landmarks = []
        if not point_cloud:
            return landmarks

        # Step 1: Simple sequential clustering of points based on distance
        clusters = []
        if point_cloud:
            current_cluster = [point_cloud[0]]
            for i in range(1, len(point_cloud)):
                p1 = point_cloud[i - 1]
                p2 = point_cloud[i]
                dist = np.sqrt(
                    (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
                )
                if dist < 0.1:  # Proximity threshold to group points into a cluster
                    current_cluster.append(p2)
                else:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = [p2]
            if current_cluster:
                clusters.append(current_cluster)

        # Step 2: Filter clusters and calculate their center to form landmarks
        for cluster in clusters:
            if self.min_points <= len(cluster) <= self.max_points:
                x_coords = [p.x for p in cluster]
                y_coords = [p.y for p in cluster]
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)

                # Further filter by the spatial size of the cluster
                max_r_sq = 0
                for p in cluster:
                    dist_sq = (p.x - center_x) ** 2 + (p.y - center_y) ** 2
                    if dist_sq > max_r_sq:
                        max_r_sq = dist_sq

                if np.sqrt(max_r_sq) < self.max_cluster_radius:
                    # This is a valid landmark.
                    # Its position (center_x, center_y) is relative to the robot.
                    # Convert to a range and bearing measurement for the EKF.
                    landmark_range = np.sqrt(center_x**2 + center_y**2)
                    landmark_bearing = np.arctan2(center_y, center_x)

                    z_k = np.array([[landmark_range], [landmark_bearing]])
                    landmarks.append(z_k)

        return landmarks
