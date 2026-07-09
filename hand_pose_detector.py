# hand_pose_detector.py
import os
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def create_hand_detector():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hand_model_path = os.path.join(script_dir, "hand_landmarker.task")
    with open(hand_model_path, "rb") as f:
        hand_model_buf = f.read()

    hand_opt = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=hand_model_buf),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    return HandLandmarker.create_from_options(hand_opt)

def create_pose_detector():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pose_model_path = os.path.join(script_dir, "pose_landmarker_lite.task")
    with open(pose_model_path, "rb") as f:
        pose_model_buf = f.read()

    pose_opt = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=pose_model_buf),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    return PoseLandmarker.create_from_options(pose_opt)
