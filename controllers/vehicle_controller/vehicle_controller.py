from rosbot import Rosbot
from apriltag_pose import AprilTagPoseEstimator


def run(robot: Rosbot) -> None:
    width = robot.camera.getWidth()
    height = robot.camera.getHeight()
    fov = robot.camera.getFov()
    est = AprilTagPoseEstimator(width, height, fov)
    while robot.step(robot.timestep) != -1:
        buf = robot.camera.getImage()
        if buf:
            tag_poses = est.estimate_pose(buf)  # update every step
    
            # Example: print a concise summary; comment out to reduce spam
            if tag_poses:
                print(f"Detected {len(tag_poses)} tag(s):")
                for d in tag_poses:
                    x, y, z = d['translation']
                    print(f"  Tag {d['tag_id']}: x={x:.3f} y={y:.3f} z={z:.3f} m  dist={d['distance']:.3f} m")
 
if __name__ == "__main__":
    robot: Rosbot = Rosbot()
    run(robot=robot)
