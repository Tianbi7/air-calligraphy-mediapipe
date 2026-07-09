# config.py
import os

# 画布分辨率
WIDTH = 640
HEIGHT = 480
w, h = WIDTH, HEIGHT

# 保存目录
SAVE_DIR = "output_calligraphy"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
SAVE_COUNTER = 0

# 宣纸模式开关
PURE_PAPER_MODE = True

# 笔迹缓存点数上限
MAX_TRACE_POINTS = 20

# 自动清屏延时（秒）
AUTO_CLEAR_DELAY = 5

# 双手张开换墨触发延时
PALM_DELAY = 2

# 墨色配置 BGR + 中文名称
INK_COLORS = [(0, 0, 0), (100, 100, 100), (150, 30, 30)]
INK_NAMES = ["浓黑墨", "淡灰墨", "朱砂红墨"]
COLOR_IDX = 0

# 书写手指标记
CURRENT_WRITING_FINGER = "index"

# 闲置计时
LAST_ACTIVE_TIME = 0.0

# 换墨动画缓存
DIP_ANIMATION = None

# 手掌状态缓存
PALM_START_TIME = None
BOTH_PALMS_OPEN = False
WAS_PINCH = False

# 姿态全局缓存
POSE_INFO = {
    "detected": False,
    "body_distance": 0,
    "body_angle": 0,
    "hands_together": False,
    "lean_forward": False
}
