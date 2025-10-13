#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import time


class Debugger:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.img_plot = None

    def debug(self, text=None, value=None):
        print(f"---> {text} : {value} <---")

    def plot_knn_match(self, pf_image, pf_kp, cf_image, cf_kp, matches):
        img3 = cv2.drawMatchesKnn(
            pf_image,
            pf_kp,
            cf_image,
            cf_kp,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        self.plot_image("", img3)

    def plot_match(self, pf_image, pf_kp, cf_image, cf_kp, matches):
        img3 = cv2.drawMatches(
            pf_image,
            pf_kp,
            cf_image,
            cf_kp,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        self.plot_image("", img3)

    def plot_image(self, title, image_array):
        if self.img_plot is None:
            self.img_plot = self.ax.imshow(image_array)
        else:
            self.img_plot.set_data(image_array)

        plt.title(title)
        plt.pause(0.01)  # Pause to allow Matplotlib to update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def sleeper(self, timer):
        time.sleep(timer)
