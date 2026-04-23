import numpy as np

class PoseProcessor:
    def __init__(self, pose):
        self.pose = pose

    def get_keypoints(self):
        keypoints = self.pose
        return np.array(keypoints).reshape(-1, 3)

    def get_confidence_scores(self):
        keypoints = self.get_keypoints()
        return keypoints[:, 2]  # Assuming the confidence score is the third element

    def filter_keypoints_by_confidence(self, threshold=0.5):
        keypoints = self.get_keypoints()
        confidence_scores = self.get_confidence_scores()
        filtered_keypoints = keypoints[confidence_scores >= threshold]
        return filtered_keypoints
    
    def angle_between_joints(self, joint1, joint2, joint3):
        # Calculate the angle between three joints
        a = np.array(joint1[:2])  # First joint
        b = np.array(joint2[:2])  # Second joint (vertex)
        c = np.array(joint3[:2])  # Third joint

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)