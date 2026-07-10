import cv2
from src.hand_pose_detector import HandDetector

def test_detector_init():
    """测试手部检测器正常初始化"""
    detector = HandDetector()
    assert detector is not None

def test_blank_image_input():
    """测试空白图像输入无崩溃报错"""
    detector = HandDetector()
    blank_img = cv2.zeros((640, 480, 3), dtype=cv2.CV_8U)
    result = detector.detect_hand(blank_img)
    assert result is None
