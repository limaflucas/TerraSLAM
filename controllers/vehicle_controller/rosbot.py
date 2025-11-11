from controller import Robot
from lidar_wrapper import LidarWrapper


class Rosbot(Robot):
    def __init__(self) -> None:
        super().__init__()

        self.timestep: int = int(self.getBasicTimeStep())
        self.camera = self.getDevice("camera rgb")
        print(self.timestep)
        self.camera.enable(self.timestep)

        self.rear_left_motor = self.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.getDevice("rr_wheel_joint")
        self.front_left_motor = self.getDevice("fl_wheel_joint")
        self.rear_right_motor = self.getDevice("fr_wheel_joint")

        self.rear_left_motor.setPosition(float("inf"))
        self.rear_right_motor.setPosition(float("inf"))
        self.front_left_motor.setPosition(float("inf"))
        self.rear_right_motor.setPosition(float("inf"))

        self.rear_left_motor.setVelocity(0.0)
        self.rear_right_motor.setVelocity(0.0)
        self.front_left_motor.setVelocity(0.0)
        self.rear_right_motor.setVelocity(0.0)

        self.lidar = self.getDevice("Hokuyo UTM-30LX")
        self.lidar_wrapper: LidarWrapper = LidarWrapper()
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.distance_sensors = [self.getDevice("fl_range"), self.getDevice("fr_range")]
        for i in self.distance_sensors:
            i.enable(self.timestep)

    def getDistanceData(self) -> None:
        r = []
        for i in self.distance_sensors:
            r.append(i.getValue())
        print(f"Distances: {r}")
