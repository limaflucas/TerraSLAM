# controllers/pioneer_controller/pioneer_controller.py
from controller import Robot
import time
import math
import cv2


# maximal speed allowed
MAX_SPEED = 10

# how many sensors are on the robot
MAX_SENSOR_NUMBER = 16

# delay used for the blinking leds
DELAY = 70

# maximal value returned by the sensors
MAX_SENSOR_VALUE = 1024.0

# minimal distance, in meters, for an obstacle to be considered
MIN_DISTANCE = 0.8

# minimal weight for the robot to turn
WHEEL_WEIGHT_THRESHOLD = 100.0


class SensorData:
    def __init__(self, wheel_weight):
        self.device = None
        self.wheel_weight = wheel_weight


# enum to represent the state of the robot
FORWARD, LEFT, RIGHT = 0, 1, 2

# how much each sensor affects the direction of the robot
sensors = [
    SensorData([150, 0]),
    SensorData([200, 0]),
    SensorData([300, 0]),
    SensorData([600, 0]),
    SensorData([0, 600]),
    SensorData([0, 300]),
    SensorData([0, 200]),
    SensorData([0, 150]),
    SensorData([0, 0]),
    SensorData([0, 0]),
    SensorData([0, 0]),
    SensorData([0, 0]),
    SensorData([0, 0]),
    SensorData([0, 0]),
    SensorData([0, 0]),
    SensorData([0, 0]),
]

red_house = {"name": "red_house", "position": (20.0, 0.0), "status": 1}
adv_board = {"name": "advertising board", "position": (-11.54, 19.79), "status": 1}
barn = {"name": "barn", "position": (-2.28, -44.03), "status": 1}
ubuilding = {"name": "u_building", "position": (-47.8, -12.8), "status": 1}
landmarks = [red_house, adv_board, barn, ubuilding]


### Calculates the robot's distance to all landmarks
### Returns a map with the distance in meters
def calculate_landmark_distance(robot_coordinates=()):
    distances = {}
    for l in landmarks:
        l_x = l["position"][0]
        l_y = l["position"][1]
        x = (l_x - robot_coordinates[0]) ** 2.0
        y = (l_y - robot_coordinates[1]) ** 2.0
        d = math.sqrt(x + y)

        lname = l["name"]
        distances[lname] = d

    return distances


def get_furthest_landmark(l_distances={}):
    if not l_distances:
        return None
    return max(l_distances, key=l_distances.get)


def main():

    try:
        print("Successfully imported OpenCV version:", cv2.__version__)
    except AttributeError:
        print("Error: OpenCV (cv2) is not installed or accessible.")

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    # Wheels
    left_wheel = robot.getDevice("left wheel")
    right_wheel = robot.getDevice("right wheel")
    left_wheel.setPosition(float("inf"))
    right_wheel.setPosition(float("inf"))
    left_wheel.setVelocity(0.0)
    right_wheel.setVelocity(0.0)

    # LEDs
    red_led = [
        robot.getDevice("red led 1"),
        robot.getDevice("red led 2"),
        robot.getDevice("red led 3"),
    ]
    for led in red_led:
        led.set(0)

    # Distance sensors
    for i in range(MAX_SENSOR_NUMBER):
        sensor_name = f"so{i}"
        sensors[i].device = robot.getDevice(sensor_name)
        sensors[i].device.enable(time_step)

    # LiDAR (optional)
    lidar = robot.getDevice("RPlidar A2")
    if lidar is not None:
        lidar.enable(time_step)
        # enablePointCloud is optional; call if available
        if hasattr(lidar, "enablePointCloud"):
            lidar.enablePointCloud()
        print(
            "LiDAR enabled: horizontalResolution =",
            lidar.getHorizontalResolution(),
            "layers =",
            lidar.getNumberOfLayers(),
        )
    else:
        print("LiDAR not found; continuing without it.")

    # camera (JetBotRaspberryPiCamera)
    try:
        camera = robot.getDevice("camera")
        camera.enable(time_step)
        print("Camera enabled:", camera.getName())
    except Exception as e:
        print("Camera not found or could not be enabled:", e)
        camera = None

    try:
        gps = robot.getDevice("gps")
        gps.enable(time_step)
        print("GPS enabled:", gps.getName())
    except Exception as e:
        print("GPS not found or could not be enabled:", e)
        gps = None

    led_number = 0
    delay = 0
    state = FORWARD
    step_count = 0

    while robot.step(time_step) != -1:
        # LiDAR reads (if available)
        if lidar is not None:
            ranges = lidar.getRangeImage()
            # if step_count % 20 == 0 and ranges:
            #     print("ranges[0..9]:", ranges[:10])
            print("lidar operational")
            # if hasattr(lidar, "getPointCloud"):
            #     points = lidar.getPointCloud()
            #     if points:
            #         p = points[0]
            #         if step_count % 20 == 0:
            #             print(
            #                 "first point: x={:.3f} y={:.3f} z={:.3f} layer={}"
            #                 .format(p.x, p.y, p.z, getattr(p, "layer_id", -1))
            #             )

        if camera is not None:
            cimage = camera.getImage()
        position = gps.getValues()
        rc = (position[0], position[1])
        landmarks_distance = calculate_landmark_distance(rc)
        l_furthest = get_furthest_landmark(landmarks_distance)

        # if cycles >= steps_per_cycle:
        # cycles = 0
        print(f"Robots coordinates: {rc[0]}:{rc[1]}")
        print(f"{landmarks_distance}")
        print(
            f"Furthest landmark is {l_furthest} at {landmarks_distance[l_furthest]} meters"
        )

        # initialize speed and wheel_weight_total at each loop
        speed = [0.0, 0.0]
        wheel_weight_total = [0.0, 0.0]

        for i in range(MAX_SENSOR_NUMBER):
            sensor_value = sensors[i].device.getValue()

            # if the sensor doesn't see anything, we don't use it this round
            if sensor_value == 0.0:
                speed_modifier = 0.0
            else:
                # computes the actual distance to the obstacle (simple LUT inverse)
                distance = 5.0 * (1.0 - (sensor_value / MAX_SENSOR_VALUE))

                # if the obstacle is close enough, influence turning
                if distance < MIN_DISTANCE:
                    speed_modifier = 1.0 - (distance / MIN_DISTANCE)
                else:
                    speed_modifier = 0.0

            # add the modifier for both wheels
            for j in range(2):
                wheel_weight_total[j] += sensors[i].wheel_weight[j] * speed_modifier

        # simplistic state machine to handle the direction of the robot
        if state == FORWARD:
            if wheel_weight_total[0] > WHEEL_WEIGHT_THRESHOLD:
                speed[0] = 0.7 * MAX_SPEED
                speed[1] = -0.7 * MAX_SPEED
                state = LEFT
            elif wheel_weight_total[1] > WHEEL_WEIGHT_THRESHOLD:
                speed[0] = -0.7 * MAX_SPEED
                speed[1] = 0.7 * MAX_SPEED
                state = RIGHT
            else:
                speed[0] = MAX_SPEED
                speed[1] = MAX_SPEED
        elif state == LEFT:
            if (
                wheel_weight_total[0] > WHEEL_WEIGHT_THRESHOLD
                or wheel_weight_total[1] > WHEEL_WEIGHT_THRESHOLD
            ):
                speed[0] = 0.7 * MAX_SPEED
                speed[1] = -0.7 * MAX_SPEED
            else:
                speed[0] = MAX_SPEED
                speed[1] = MAX_SPEED
                state = FORWARD
        elif state == RIGHT:
            if (
                wheel_weight_total[0] > WHEEL_WEIGHT_THRESHOLD
                or wheel_weight_total[1] > WHEEL_WEIGHT_THRESHOLD
            ):
                speed[0] = -0.7 * MAX_SPEED
                speed[1] = 0.7 * MAX_SPEED
            else:
                speed[0] = MAX_SPEED
                speed[1] = MAX_SPEED
                state = FORWARD

        # LED blinking
        delay += 1
        if delay == DELAY:
            red_led[led_number].set(0)
            led_number = (led_number + 1) % 3
            red_led[led_number].set(1)
            delay = 0

        # set motor speeds
        left_wheel.setVelocity(speed[0])
        right_wheel.setVelocity(speed[1])

        step_count += 1


if __name__ == "__main__":
    main()
