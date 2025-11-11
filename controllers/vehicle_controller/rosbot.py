from controller import Robot
from lidar_wrapper import LidarWrapper


class Rosbot(Robot):
    def __init__(self) -> None:
        super().__init__()

        self.timestep: int = int(self.getBasicTimeStep())
        self.camera = self.getDevice("camera")
        print(self.timestep)
        self.camera.enable(self.timestep)

        self.left_wheel = self.getDevice("left wheel")
        self.right_wheel = self.getDevice("right wheel")

        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))

        self.left_wheel.setVelocity(0.0)
        self.right_wheel.setVelocity(0.0)

        self.lidar = self.getDevice("Hokuyo UTM-30LX")
        self.lidar_wrapper: LidarWrapper = LidarWrapper()
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # self.distance_sensors = [self.getDevice("fl_range"), self.getDevice("fr_range")]
        # for i in self.distance_sensors:
        #     i.enable(self.timestep)

    def getDistanceData(self) -> None:
        r = []
        for i in self.distance_sensors:
            r.append(i.getValue())
        print(f"Distances: {r}")
