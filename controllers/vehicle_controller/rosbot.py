from controller import Robot # type: ignore
from lidar_wrapper import LidarWrapper


class Rosbot(Robot):
    def __init__(self) -> None:
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())

         # List available devices to verify names
        for i in range(self.getNumberOfDevices()):
            d = self.getDeviceByIndex(i)
            print("Device:", d.getName())

        # Camera (Astra color stream)
        self.camera = self.getDevice("camera rgb")
        assert self.camera is not None, "Device 'camera rgb' not found"
        self.camera.enable(self.timestep)

        self.left_wheel  = self.getDevice("fl_wheel_joint")   # front-left
        self.right_wheel = self.getDevice("fr_wheel_joint")   # front-right
        assert self.left_wheel is not None,  "Device 'fl_wheel_joint' not found"
        assert self.right_wheel is not None, "Device 'fr_wheel_joint' not found"

        self.left_wheel.setPosition(float("inf"))
        self.right_wheel.setPosition(float("inf"))

        self.left_wheel.setVelocity(0.0)
        self.right_wheel.setVelocity(0.0)

        self.lidar = self.getDevice("laser")
        assert self.lidar is not None, "Device 'laser' not found"
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
