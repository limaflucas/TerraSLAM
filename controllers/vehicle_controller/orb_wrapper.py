#!/usr/bin/env python3
import cv2
import numpy as np


### ORB frames control
class ORBWrapper:
    def __init__(self, image_height, image_width, focal):
        self._orb = cv2.ORB_create(nfeatures=2000)

        # Initialize Matcher (Brute-Force Matcher with NORM_HAMMING for binary descriptors like ORB)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # k best matches
        self._k = 2

        # ration test to accept or reject matches
        self._ratio_test = 0.75

        self._image_height = image_height
        self._image_width = image_width
        self._focal = focal
        self._principal_point = (
            self._image_width / 2,
            self._image_height / 2,
        )
        self.current_frame = None
        self.previous_frame = None

    def compute(self, image):
        if image is None:
            raise Exception("Missing image to be converted")

        np_image = self._convert_image_to_bgr(image)
        # Convert BGR image to grayscale for ORB detection
        grey_scale = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        keyframe, descriptors = self._orb.detectAndCompute(grey_scale, None)
        self._place_new_frame(keyframe, descriptors)
        return self._match_frames()

    def estimate_range_bearing(self, kp_pframe, kp_cframe, R, t):
        """
        REAL Measurement Model: Triangulates the 3D position and converts it
        to a Range and Bearing measurement (z_k).
        """

        # Triangulate the 3D position of the landmark relative to the PREVIOUS frame's camera center
        # NOTE: We use the first point [0] for simplicity, as only ONE z_k is passed to EKF
        point_3d_c1 = self.triangulate_3d_point(kp_pframe[0], kp_cframe[0], R, t)

        # Since the 3D point is calculated in the camera frame, we assume the camera
        # is aligned with the robot's local frame (x-forward, y-up, z-right, or similar).
        # For a Pioneer robot in Webots, the coordinates are (X, Y, Z) where Z is forward.
        # The Lidar and 2D SLAM use (X, Z) where X is forward. We'll use the 2D plane (X, Z)
        # where X is the forward/range component and Z is the lateral/width component.

        # 1. Range (r): Euclidean distance from the origin (camera center) to the 3D point.
        # Assuming camera frame axes are [X_fwd, Y_up, Z_lat] for simplification:
        X_fwd = point_3d_c1[0, 0]
        Y_up = point_3d_c1[1, 0]  # Often ignored in 2D SLAM
        Z_lat = point_3d_c1[2, 0]

        r = np.sqrt(X_fwd**2 + Z_lat**2)

        # 2. Bearing (phi): Angle from the camera's forward axis (X) to the point in the XZ plane.
        # phi = atan2(Z_lat, X_fwd)
        phi = np.arctan2(Z_lat, X_fwd)

        # Add small sensor noise (as is required for testing EKF's Q-matrix)
        r += np.random.normal(0, 0.05)
        phi += np.random.normal(0, 0.01)

        # Return z_k as a 2x1 NumPy column vector: [[r], [phi]]
        return np.array([[r], [phi]])

    def triangulate_3d_point(self, kp1, kp2, R, t):
        """
        Triangulates the 3D position of a feature using two matched 2D points
        and the relative camera pose (R, t).

        kp1 and kp2 must be single points of shape (1, 2).
        Returns: 3D point (numpy array 3x1)
        """

        # --- Camera Intrinsics (Must be accurate for your Webots camera) ---
        # We assume a simple pinhole model: fx = fy = focal_length, cx, cy at center
        fx = self._focal
        fy = self._focal  # fx = fy for square pixels
        cx = self._image_width / 2.0
        cy = self._image_height / 2.0

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Projection Matrix P1 (Identity pose)
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

        # Projection Matrix P2 (Relative pose)
        P2 = K @ np.hstack((R, t))

        # Triangulate the point
        point_4d = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)

        # Convert from homogeneous (4D) to Euclidean (3D) coordinates
        point_3d_euclidean = point_4d[:3] / point_4d[3]

        return point_3d_euclidean  # Returns 3x1 column vector: [X_c, Y_c, Z_c] relative to camera 1

    # --- Step 1: Recover Pose from ORB Matches (Visual Odometry) ---
    # Placeholder for Camera Intrinsics (K) - Use the same simplified values as Triangulation
    def get_relative_pose(self, kp_pframe, kp_cframe):
        # Ensure keypoint arrays are in a contiguous memory layout for OpenCV
        kp_pframe_cont = np.ascontiguousarray(kp_pframe)
        kp_cframe_cont = np.ascontiguousarray(kp_cframe)

        # Find the Essential Matrix (E) using RANSAC
        E, mask = cv2.findEssentialMat(
            kp_pframe_cont,
            kp_cframe_cont,
            focal=self._focal,
            pp=self._principal_point,
            method=cv2.RANSAC,
            threshold=1.0,
        )  # Pixel error threshold
        #
        # Recover the relative pose (Rotation R and Translation t)
        _, R, t, _ = cv2.recoverPose(
            E,
            kp_pframe_cont,
            kp_cframe_cont,
            focal=self._focal,
            pp=self._principal_point,
            mask=mask,
        )

        return R, t

    def _match_frames(self):
        if self.previous_frame is None or self.current_frame is None:
            return None

        matches = self._matcher.knnMatch(
            self.previous_frame.descriptors, self.current_frame.descriptors, self._k
        )

        above_ratio_matches = []
        for m, n in matches:
            if m.distance < self._ratio_test * n.distance:
                above_ratio_matches.append(m)

        if len(above_ratio_matches) < 8:
            return None

        # Retrive keypoints from previous frame
        kp_pframe = np.float32(
            [self.previous_frame.keyframes[m.queryIdx].pt for m in above_ratio_matches]
        ).reshape(-1, 1, 2)
        # Retrive keypoints from current frame
        kp_cframe = np.float32(
            [self.current_frame.keyframes[m.trainIdx].pt for m in above_ratio_matches]
        ).reshape(-1, 1, 2)
        # Retrieve descriptors for current frame
        # Will be used in the data association step
        desc_cframe = np.float32(
            [self.current_frame.descriptors[m.trainIdx] for m in above_ratio_matches]
        )

        return ORBResult(kp_pframe, kp_cframe, desc_cframe, above_ratio_matches)

    # Frame control that is used to calculate keyfeatures
    def _place_new_frame(self, kf, desc):
        self.previous_frame = self.current_frame
        self.current_frame = ComputedORB(kf, desc)

    # it receives the encoded image as B, G, R, Alpha, to convert to an 2D-array with the B, G, R, Alpha as data
    # Returns only the 2D- array with B, G, R data
    def _convert_image_to_bgr(self, image=None):
        numpy_array = np.frombuffer(image, np.uint8).reshape(
            (self._image_height, self._image_width, 4)
        )
        return numpy_array[:, :, :3]


class ComputedORB:
    def __init__(self, kf, desc):
        self.keyframes = kf
        self.descriptors = desc


class ORBResult:
    def __init__(self, kp_pf, kp_cf, desc_cf, matches):
        self.kp_pf = kp_pf
        self.kp_cf = kp_cf
        self.desc_cf = desc_cf
        self.matches = matches
