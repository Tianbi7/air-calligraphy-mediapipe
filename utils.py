# utils.py
import cv2
import numpy as np
import mediapipe as mp

# 坐标归一化转像素
def get_point(landmark, img_w, img_h):
    return int(landmark.x * img_w), int(landmark.y * img_h)

# 拇指-食指指尖距离
def get_thumb_index_dist(landmarks):
    t = landmarks[4]
    i = landmarks[8]
    return np.sqrt((t.x - i.x) ** 2 + (t.y - i.y) ** 2)

# 判断是否右手
def is_right_hand(results, hand_index):
    if results.handedness:
        return results.handedness[hand_index][0].category_name == "Right"
    return True

# 判断手掌是否张开
def is_palm_open(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]
    open_count = 0
    for tip, mcp in zip(finger_tips, finger_mcps):
        dist = np.sqrt((landmarks[tip].x - landmarks[mcp].x) ** 2 +
                       (landmarks[tip].y - landmarks[mcp].y) ** 2)
        if dist > 0.08:
            open_count += 1
    return open_count >= 4

# 姿态数据解析
def analyze_pose(pose_results, pose_info):
    if pose_results and pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks[0]
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        shoulder_center = np.array([(left_shoulder.x + right_shoulder.x) / 2,
                                    (left_shoulder.y + right_shoulder.y) / 2])
        hip_center = np.array([(left_hip.x + right_hip.x) / 2,
                               (left_hip.y + right_hip.y) / 2])

        body_height = np.linalg.norm(shoulder_center - hip_center)
        pose_info["body_distance"] = body_height

        spine_vector = hip_center - shoulder_center
        pose_info["body_angle"] = np.degrees(np.arctan2(spine_vector[0], spine_vector[1]))

        wrist_dist = np.sqrt((left_wrist.x - right_wrist.x) ** 2 +
                             (left_wrist.y - right_wrist.y) ** 2)
        pose_info["hands_together"] = wrist_dist < 0.15

        face_to_shoulder = np.sqrt((nose.x - shoulder_center[0]) ** 2 +
                                   (nose.y - shoulder_center[1]) ** 2)
        pose_info["lean_forward"] = face_to_shoulder > 0.15
        pose_info["detected"] = True
    else:
        pose_info["detected"] = False

# 相机初始化兼容多后端
def init_camera(target_w, target_h):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            # 预热帧
            for _ in range(5):
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    return cap
            cap.release()
    return None
