"""
apriltag_pose.py
AprilTag pose estimation using pupil-apriltags (Windows-compatible)
"""

import cv2
import numpy as np
import math
from pupil_apriltags import Detector


class AprilTagPoseEstimator:
    def __init__(self, camera_width, camera_height, camera_fov, tag_size=0.16):
        """
        Initialize the AprilTag pose estimator

        Args:
            camera_width: Camera image width in pixels
            camera_height: Camera image height in pixels
            camera_fov: Camera field of view in radians
            tag_size: Physical size of AprilTag in meters (default 0.16m)
        """
        self.width = camera_width
        self.height = camera_height
        self.cx = camera_width / 2.0
        self.cy = camera_height / 2.0
        self.fx = (camera_width / 2.0) / math.tan(camera_fov / 2.0)
        self.fy = self.fx
        self.camera_params = (self.fx, self.fy, self.cx, self.cy)
        self.tag_size = tag_size

        # Create pupil-apriltags detector
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
            debug=False,
        )

        print(f"AprilTag detector initialized:")
        print(
            f"  Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
        )
        print(f"  Tag size: {self.tag_size}m")

    def estimate_pose(self, camera_image):
        """
        Detect AprilTags and estimate their poses

        Args:
            camera_image: Raw image buffer from Webots camera

        Returns:
            List of dicts: [{'tag_id': int, 'translation': [x, y, z], 'rotation': 3x3, 'distance': float}, ...]
        """
        # Convert Webots BGRA image to grayscale
        image = np.frombuffer(camera_image, np.uint8).reshape(
            (self.height, self.width, 4)
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        # Detect tags with pose estimation
        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

        detections = []
        for r in results:
            translation = r.pose_t.flatten()
            rotation = r.pose_R
            distance = np.linalg.norm(translation)

            detections.append(
                {
                    "tag_id": r.tag_id,
                    "translation": translation,
                    "rotation": rotation,
                    "distance": distance,
                }
            )

        return detections
