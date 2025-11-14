from controller import Robot


class PioneerDX(Robot):
    def __init__(self, lidar_noise: float = 0.0) -> None:
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

        self.lidar = self.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        self.lidar.noise = lidar_noise
        self.lidar.enablePointCloud()
