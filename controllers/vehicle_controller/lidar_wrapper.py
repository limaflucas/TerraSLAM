#!/usr/bin/env python3
from typing import Any
import numpy as np
import math


class LidarWrapper:
    def __init__(self, min_points=4, max_points=5, max_cluster_radius=0.2):
        self.min_points = min_points
        self.max_points = max_points
        self.max_cluster_radius = max_cluster_radius

    def get_largest_cluster(self, point_cloud: list[Any]) -> list[Any]:
        """
        Given all the points from a LIDAR sensor:
        1. Find the distance between points
        2. Cluster the points with max_cluster_radius
        3. Return the cluster with most points
        """
        if not point_cloud:
            print(f"No points to be clustered")
            return []

        pre_clusters = []
        local_cluster = [point_cloud[0]]
        for i in range(1, len(point_cloud)):
            p1 = point_cloud[i - 1]
            p2 = point_cloud[i]
            dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
            if dist < self.max_cluster_radius:
                local_cluster.append(p2)
            else:
                if local_cluster:
                    pre_clusters.append(local_cluster)
                local_cluster = [p2]
        if local_cluster:
            pre_clusters.append(local_cluster)

        largest_cluster = []
        for c in pre_clusters:
            if len(c) > len(largest_cluster):
                largest_cluster = c

        return largest_cluster

    def get_distance(self, cluster: list[Any]) -> float | None:
        """
        Calculating the Euclidean distancce to the projected center of the points
        """
        x_coords = [p.x for p in cluster]
        y_coords = [p.y for p in cluster]  # 2D SLAM often ignores z

        # Find the center of the object
        center_x = sum(x_coords) / len(cluster)
        center_y = sum(y_coords) / len(cluster)

        # Calculate the distance to that center
        distance: float = math.sqrt(center_x**2 + center_y**2)

        print(f"The object is {distance:.3f} meters away.")

        return distance
