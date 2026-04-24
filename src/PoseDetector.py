import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:
    def __init__(
        self,
        mode=False,
        upBody=False,
        smooth=True,
        detectionCon=0.5,
        trackCon=0.5,
        model_path="model/pose_landmarker_full.task",
    ):
        self.model_path = model_path
        self._timestamp_ms = 0

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def findPose(self, img, timestamp_ms, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        if timestamp_ms is not None:
            self._timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if draw and result.pose_landmarks:
            self._draw(img, result)
        return img, result

    @staticmethod
    def _draw(img, result):
        h, w = img.shape[:2]
        for landmarks in result.pose_landmarks:
            for lm in landmarks:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
            for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
                start_idx, end_idx = connection.start, connection.end
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                cv2.line(
                    img,
                    (int(start_lm.x * w), int(start_lm.y * h)),
                    (int(end_lm.x * w), int(end_lm.y * h)),
                    (0, 255, 0),
                    2,
                )

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
