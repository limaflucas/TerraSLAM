"""vehicle_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import time
import math

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

### Landmarks
### Map with all landmarks in the map. The position field is a tupple of coordinates (x, y).
red_house = {"name": "red_house", "position": (20.0, 0.0), "status": 1}
adv_board = {"name": "advertising board", "position": (-11.54, 19.79), "status": 1}
barn = {"name": "barn", "position": (-2.28, -44.03), "status": 1}
ubuilding = {"name": "u_building", "position": (-47.8, -12.8), "status": 1}
landmarks = [red_house, adv_board, barn, ubuilding]

### Devices
lmotor = robot.getDevice("left wheel")
rmotor = robot.getDevice("right wheel")
camera = robot.getDevice("camera")
gps = robot.getDevice("gps")

### Configure Devices
lmotor.setPosition(float("inf"))
rmotor.setPosition(float("inf"))
lmotor.setVelocity(0.0)
rmotor.setVelocity(0.0)

camera.enable(timestep)
gps.enable(timestep)


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
    distance = 0
    name = ""
    for d in l_distances.keys():
        if distance < l_distances[d]:
            name = d
            distance = l_distances[d]
    return name


cycles = 99999
steps_per_cycle = int(1000 / timestep)
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    cycles += 1

    lmotor.setVelocity(1.0)
    rmotor.setVelocity(1.0)

    cimage = camera.getImage()
    position = gps.getValues()
    rc = (position[0], position[1])
    landmarks_distance = calculate_landmark_distance(rc)
    l_furthest = get_furthest_landmark(landmarks_distance)

    if cycles >= steps_per_cycle:
        cycles = 0
        print(f"Robots coordinates: {rc[0]}:{rc[1]}")
        print(f"{landmarks_distance}")
        print(
            f"Furthest landmark is {l_furthest} at {landmarks_distance[l_furthest]} meters"
        )

    # print(f"{cimage}")

    # n_devices = robot.getNumberOfDevices()
    # for i in range(n_devices):

    #   device = robot.getDeviceByIndex(i)
    #   name = device.getName()
    #   dtype = device.getNodeType()
    #   print(f"Device #{i} {device} - name: {name} type: {dtype}\n")

# Enter here exit cleanup code.
