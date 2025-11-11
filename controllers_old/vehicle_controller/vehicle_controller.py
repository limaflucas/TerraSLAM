from rosbot import Rosbot

print(f"bora rodar2")


def run(robot: Rosbot):
    print(f"bora rodar3")
    while robot.timestep != -1:
        print(f"bora rodar")


if __name__ == "__main__":
    print(f"bora rodar4")
    robot: Rosbot = Rosbot()
    run(robot=robot)
