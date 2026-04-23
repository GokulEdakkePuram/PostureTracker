import cv2 
import time

from src.PoseDetector import PoseDetector
from src.PoseProcessor import PoseProcessor
from src.DisplayUI import DisplayUI

def main():
    cv2apture = cv2.VideoCapture(0)
    pose_detector = PoseDetector()
    display_ui = DisplayUI()
    while True:
        success, img = cv2apture.read()
        timestamp_ms = int(time.time() * 1000)
        if not success:
            break

        img, result = pose_detector.findPose(img, timestamp_ms)
        if result.pose_landmarks:
            pose_processor = PoseProcessor(result.pose_landmarks[0])
            keypoints = pose_processor.get_keypoints()
            # confidence_scores = pose_processor.get_confidence_scores()
            # filtered_keypoints = pose_processor.filter_keypoints_by_confidence()
            filtered_keypoints = keypoints  # For simplicity, using all keypoints without filtering

            # Example: Calculate angle between three joints (e.g., shoulder, elbow, wrist)
            # if len(filtered_keypoints) >= 3:
            #     angle = pose_processor.angle_between_joints(filtered_keypoints[0], 
            #                                                 filtered_keypoints[1], 
            #                                                 filtered_keypoints[2])
            #     display_ui.display_angle(img, angle)

        display_ui.show(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
