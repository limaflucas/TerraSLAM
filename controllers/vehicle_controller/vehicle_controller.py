from ekf import EKF
from lidar_wrapper import LidarWrapper
from pioneer_robot import PioneerDX
import numpy as np
import time


def run(robot: PioneerDX, lidar_wrapper: LidarWrapper, ekf: EKF) -> None:
    while robot.step(robot.timestep) != -1:
        pc: bytearray = robot.lidar.getPointCloud()
        cluster = lidar_wrapper.get_largest_cluster(pc)
        distance: float | None = lidar_wrapper.get_distance(cluster)
        print(f"distance: {distance}")
        time.sleep(2)


if __name__ == "__main__":
    robot: PioneerDX = PioneerDX()
    lidar_wrapper: LidarWrapper = LidarWrapper()
    ekf_wrapper: EKF = EKF()
    run(robot=robot, lidar_wrapper=lidar_wrapper, ekf=ekf)
