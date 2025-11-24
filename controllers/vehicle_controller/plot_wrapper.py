import matplotlib.pyplot as plt
from collections import deque
import numpy as np


class SLAMGraph:
    def __init__(self, event) -> None:
        plt.ion()
        self.fig, (self.ax_traj, self.ax_pose, self.ax_heading) = plt.subplots(
            3, 1, figsize=(8, 12), constrained_layout=True
        )

        self.fig.canvas.mpl_connect("key_press_event", event)

        # Trajectory plot
        (self.line_gt,) = self.ax_traj.plot(
            [], [], "g.", label="Ground Truth (GPS)", markersize=2
        )
        (self.line_prediction,) = self.ax_traj.plot(
            [], [], "k.", label="EKF Prediction", markersize=1, alpha=0.6
        )
        (self.line_correction,) = self.ax_traj.plot(
            [], [], "r.", label="EKF Correction", markersize=4
        )
        self.ax_traj.set_title("SLAM Trajectory Analysis")
        self.ax_traj.set_xlabel("X Position (m)")
        self.ax_traj.set_ylabel("Y Position (m)")
        self.ax_traj.grid(True)
        self.ax_traj.legend(
            bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0.1
        )
        self.ax_traj.set_aspect("equal")

        # Pose error plot
        (self.line_pose_err,) = self.ax_pose.plot([], [], "b-", label="Pose Error (m)")
        self.ax_pose.set_title("Pose Error Evolution")
        self.ax_pose.set_ylabel("Error (meters)")
        self.ax_pose.grid(True)
        self.ax_pose.legend(loc="upper right")

        # Heading error plot
        (self.line_heading_err,) = self.ax_heading.plot(
            [], [], "r-", label="Heading Error (rad)"
        )

        self.ax_heading.set_title("Heading Error Evolution")
        self.ax_heading.set_xlabel("Time (s)")
        self.ax_heading.set_ylabel("Error (radians)")
        self.ax_heading.grid(True)
        self.ax_heading.legend(loc="upper right")

        MAX_POINTS = 5000
        # Trajectory data
        self.hist_gt_x, self.hist_gt_y = deque(maxlen=MAX_POINTS), deque(
            maxlen=MAX_POINTS
        )
        self.hist_prediction_x, self.hist_prediction_y = deque(
            maxlen=MAX_POINTS
        ), deque(maxlen=MAX_POINTS)
        self.hist_correction_x, self.hist_correction_y = deque(
            maxlen=MAX_POINTS
        ), deque(maxlen=MAX_POINTS)

        # Error evolution data
        self.time_history = deque(maxlen=MAX_POINTS)
        self.pose_err_history = deque(maxlen=MAX_POINTS)
        self.heading_err_history = deque(maxlen=MAX_POINTS)

    def append_data_ground_truth(self, x, y):
        self.hist_gt_x.append(x)
        self.hist_gt_y.append(y)

    def append_data_prediction(self, x, y):
        self.hist_prediction_x.append(x)
        self.hist_prediction_y.append(y)

    def append_data_ekf_correction(self, x, y):
        self.hist_correction_x.append(x)
        self.hist_correction_y.append(y)

    def append_data_time(self, time):
        self.time_history.append(time)

    def append_data_pose(self, gps_x, gps_y, ekf_x, ekf_y):
        dx = gps_x - ekf_x
        dy = gps_y - ekf_y
        distance = np.sqrt(dx**2 + dy**2)
        self.pose_err_history.append(distance)

    def append_data_heading(self, compass_x, compass_y, ekf_theta):
        compass_theta = np.atan2(compass_y, compass_x)
        diff = compass_theta - ekf_theta
        heading = np.atan2(np.sin(diff), np.cos(diff))
        self.heading_err_history.append(heading)

    def draw(self):
        # Update trajectory
        self.line_gt.set_data(self.hist_gt_x, self.hist_gt_y)
        self.line_prediction.set_data(self.hist_prediction_x, self.hist_prediction_y)
        self.line_correction.set_data(self.hist_correction_x, self.hist_correction_y)

        # Update errors
        self.line_pose_err.set_data(self.time_history, self.pose_err_history)
        self.line_heading_err.set_data(self.time_history, self.heading_err_history)

        # Autoscale all axes
        for ax in [self.ax_traj, self.ax_pose, self.ax_heading]:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
