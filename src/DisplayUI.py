import cv2

# import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision


class DisplayUI:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.window_name = "Pose Detection"
        cv2.namedWindow(self.window_name)

    def show(self, img):
        cv2.imshow(self.window_name, img)

    def close(self):
        cv2.destroyAllWindows()

    def display_angle(self, img, joint_name, angle, position=(50, 50)):
        cv2.putText(
            img,
            f"{joint_name}: {angle:.2f}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

    def draw_landmarks(self, img, landmarks):
        pose_connection_style = drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=7, circle_radius=5
        )
        drawing_utils.draw_landmarks(
            img,
            landmarks,
            vision.PoseLandmarkersConnections.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=pose_connection_style,
        )
