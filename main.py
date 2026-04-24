import cv2
import time
import numpy as np
import threading

from src.PoseDetector import PoseDetector
from src.PoseProcessor import PoseProcessor
from src.DisplayUI import DisplayUI
from src.constants import POSE_LANDMARK_INDEXES


class PostureTracker:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.display_ui = DisplayUI()
        self.image = np.zeros(
            (self.display_ui.height, self.display_ui.width, 3), dtype=np.uint8
        )
        self.timestamp_ms = 0
        self.frame_lock = threading.Lock()
        self.stop_capture = threading.Event()

    def capture_frames(self):
        capture = cv2.VideoCapture(0)
        while not self.stop_capture.is_set():
            success, img = capture.read()
            self.timestamp_ms = int(time.time() * 1000)
            if not success:
                break
            with self.frame_lock:
                self.image = img.copy()

    def get_img(self):
        with self.frame_lock:
            return self.image, self.timestamp_ms

    def close(self):
        self.stop_capture.set()
        self.pose_detector.close()
        self.display_ui.close()


def main():
    posture_tracker = PostureTracker()
    frame_reader_thread = threading.Thread(
        target=posture_tracker.capture_frames, daemon=True
    )
    frame_reader_thread.start()
    img, timestamp_ms = posture_tracker.get_img()

    while img is not None:
        img, timestamp_ms = posture_tracker.get_img()
        if timestamp_ms == 0:
            print("Waiting for frames...")
            continue
        img, result = posture_tracker.pose_detector.findPose(
            img, timestamp_ms=int(time.time() * 1000), draw=True
        )
        if result.pose_landmarks:
            pose_processor = PoseProcessor(result.pose_landmarks[0])
            keypoints = pose_processor.get_keypoints()
            # confidence_scores = pose_processor.get_confidence_scores()
            # filtered_keypoints = pose_processor.filter_keypoints_by_confidence()
            # filtered_keypoints = keypoints  # For simplicity, using all keypoints without filtering

            # Example: Calculate angle between three joints (e.g., shoulder, elbow, wrist)
            # joints_of_interest = ["left shoulder", "left elbow", "left wrist", "right shoulder", "right elbow", "right wrist"]
            angles = {
                "left shoulder": pose_processor.calculate_angle(
                    keypoints[POSE_LANDMARK_INDEXES["right shoulder"]],
                    keypoints[POSE_LANDMARK_INDEXES["left shoulder"]],
                    keypoints[POSE_LANDMARK_INDEXES["left elbow"]],
                ),
                "right shoulder": pose_processor.calculate_angle(
                    keypoints[POSE_LANDMARK_INDEXES["left shoulder"]],
                    keypoints[POSE_LANDMARK_INDEXES["right shoulder"]],
                    keypoints[POSE_LANDMARK_INDEXES["right elbow"]],
                ),
                "left elbow": pose_processor.calculate_angle(
                    keypoints[POSE_LANDMARK_INDEXES["left shoulder"]],
                    keypoints[POSE_LANDMARK_INDEXES["left elbow"]],
                    keypoints[POSE_LANDMARK_INDEXES["left wrist"]],
                ),
                "right elbow": pose_processor.calculate_angle(
                    keypoints[POSE_LANDMARK_INDEXES["right shoulder"]],
                    keypoints[POSE_LANDMARK_INDEXES["right elbow"]],
                    keypoints[POSE_LANDMARK_INDEXES["right wrist"]],
                ),
            }
            # if len(filtered_keypoints) >= 3:
            #     angle = pose_processor.angle_between_joints(filtered_keypoints[0],
            #                                                 filtered_keypoints[1],
            #                                                 filtered_keypoints[2])
            for joint_name, angle in angles.items():
                posture_tracker.display_ui.display_angle(
                    img,
                    joint_name,
                    angle,
                    position=(50, 50 + 30 * list(angles.keys()).index(joint_name)),
                )
            # posture_tracker.display_ui.display_angle(img, angle)
        posture_tracker.display_ui.show(img)
        # cv2.imshow(posture_tracker.display_ui.window_name, posture_tracker.get_img()[0])
        # print("Frame displayed")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            posture_tracker.close()
            if frame_reader_thread.is_alive():
                frame_reader_thread.join()
            break


if __name__ == "__main__":
    main()
