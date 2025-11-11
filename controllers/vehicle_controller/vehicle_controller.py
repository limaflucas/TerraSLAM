from rosbot import Rosbot


def run(robot: Rosbot) -> None:
    while robot.step(robot.timestep) != -1:
        print(f"logic goes here")


if __name__ == "__main__":
    robot: Rosbot = Rosbot()
    run(robot=robot)
