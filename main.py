# main.py
import cv2
import numpy as np
import time
import mediapipe as mp

# 导入拆分模块
from config import *
from utils import get_point, is_right_hand, is_palm_open, analyze_pose, init_camera
from paper_render import create_rice_paper, ink_render_kai
from hand_pose_detector import create_hand_detector, create_pose_detector

# 初始化检测器
hand_detector = create_hand_detector()
pose_detector = create_pose_detector()

# 初始化画布
ink_canvas = create_rice_paper(h, w)
trace_points = []

# 打开相机
cap = init_camera(w, h)
if cap is None:
    print("ERROR: Cannot open camera")
    exit(1)

def main():
    global ink_canvas, trace_points, COLOR_IDX, SAVE_COUNTER, CURRENT_WRITING_FINGER
    global LAST_ACTIVE_TIME, PALM_START_TIME, BOTH_PALMS_OPEN, DIP_ANIMATION

    hand_detected = False
    is_writing = False
    idle_start = time.time()
    last_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Error", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("Air Calligraphy", frame)
                cv2.imshow("Ink Canvas", ink_canvas)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # 手部检测
            try:
                hand_results = hand_detector.detect_for_video(mp_image, timestamp_ms)
            except:
                hand_results = None

            # 姿态检测
            try:
                pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)
                analyze_pose(pose_results, POSE_INFO)
            except:
                POSE_INFO["detected"] = False

            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time

            writing_tip = None
            hand_detected = False
            right_hand_lm = None
            left_hand_lm = None
            right_palm_open = False
            left_palm_open = False

            if hand_results and hand_results.hand_landmarks:
                hand_detected = True
                for i, hand_lm in enumerate(hand_results.hand_landmarks):
                    if is_right_hand(hand_results, i):
                        right_hand_lm = hand_lm
                    else:
                        left_hand_lm = hand_lm
                if right_hand_lm is None and hand_results.hand_landmarks:
                    right_hand_lm = hand_results.hand_landmarks[0]

                # 右手绘制关键点 + 判断书写
                if right_hand_lm:
                    landmarks = right_hand_lm
                    for _, lm in enumerate(landmarks):
                        cx, cy = get_point(lm, w, h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    conns = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]]
                    for c in conns:
                        x1,y1 = get_point(landmarks[c[0]],w,h)
                        x2,y2 = get_point(landmarks[c[1]],w,h)
                        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    index_tip = get_point(landmarks[8],w,h)
                    thumb_tip = get_point(landmarks[4],w,h)
                    cv2.circle(frame,index_tip,6,(0,0,255),-1)
                    cv2.circle(frame,thumb_tip,6,(255,0,0),-1)
                    right_palm_open = is_palm_open(landmarks)

                    index_ext = np.linalg.norm(np.array([landmarks[8].x, landmarks[8].y]) - np.array([landmarks[6].x, landmarks[6].y])) > 0.05
                    thumb_ext = np.linalg.norm(np.array([landmarks[4].x, landmarks[4].y]) - np.array([landmarks[3].x, landmarks[3].y])) > 0.05
                    if thumb_ext and index_ext:
                        writing_tip = index_tip
                        CURRENT_WRITING_FINGER = "index"
                    elif index_ext:
                        writing_tip = index_tip
                        CURRENT_WRITING_FINGER = "index"
                    elif thumb_ext:
                        writing_tip = thumb_tip
                        CURRENT_WRITING_FINGER = "thumb"
                    else:
                        writing_tip = None

                # 左手绘制
                if left_hand_lm:
                    landmarks = left_hand_lm
                    for _, lm in enumerate(landmarks):
                        cx, cy = get_point(lm, w, h)
                        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)
                    conns = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]]
                    for c in conns:
                        x1,y1 = get_point(landmarks[c[0]],w,h)
                        x2,y2 = get_point(landmarks[c[1]],w,h)
                        cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),2)
                    left_palm_open = is_palm_open(landmarks)

                # 双手张开换墨逻辑
                if right_palm_open and left_palm_open:
                    if PALM_START_TIME is None:
                        PALM_START_TIME = time.time()
                    else:
                        elapse = time.time() - PALM_START_TIME
                        if elapse >= PALM_DELAY and not BOTH_PALMS_OPEN:
                            COLOR_IDX = (COLOR_IDX + 1) % len(INK_COLORS)
                            BOTH_PALMS_OPEN = True
                            DIP_ANIMATION = {"x": w//2, "y": h//2, "start": time.time()}
                else:
                    PALM_START_TIME = None
                    BOTH_PALMS_OPEN = False

            # 笔迹绘制
            if writing_tip is not None:
                LAST_ACTIVE_TIME = time.time()
                if len(trace_points) == 0:
                    trace_points.append(writing_tip)
                else:
                    last_pt = trace_points[-1]
                    dist = np.sqrt((writing_tip[0]-last_pt[0])**2 + (writing_tip[1]-last_pt[1])**2)
                    if dist > 2:
                        trace_points.append(writing_tip)
                        if len(trace_points) > MAX_TRACE_POINTS:
                            trace_points.pop(0)
                if len(trace_points) >= 2:
                    brush_size = 20 if CURRENT_WRITING_FINGER == "thumb" else 6
                    ink_canvas = ink_render_kai(ink_canvas, trace_points, brush_size, INK_COLORS[COLOR_IDX], h, w)
            else:
                trace_points.clear()
                is_writing = False

            # 自动清屏
            if time.time() - LAST_ACTIVE_TIME > AUTO_CLEAR_DELAY:
                ink_canvas = create_rice_paper(h, w)
                trace_points.clear()
                LAST_ACTIVE_TIME = time.time()

            # 自动保存画布
            if time.time() - idle_start > 5 and np.sum(255 - ink_canvas) > 500:
                save_path = os.path.join(SAVE_DIR, f"calli_{SAVE_COUNTER}.png")
                cv2.imwrite(save_path, ink_canvas)
                SAVE_COUNTER += 1
                idle_start = time.time()

            # 画布融合
            if PURE_PAPER_MODE:
                frame_with_ink = ink_canvas.copy()
            else:
                ink_gray = cv2.cvtColor(ink_canvas, cv2.COLOR_BGR2GRAY)
                _, ink_mask = cv2.threshold(ink_gray, 245, 255, cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(ink_mask)
                bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                fg = cv2.bitwise_and(ink_canvas, ink_canvas, mask=ink_mask)
                frame_with_ink = cv2.add(bg, fg)

            # 换墨动画绘制
            if DIP_ANIMATION:
                elapse = time.time() - DIP_ANIMATION["start"]
                if elapse < 1.5:
                    progress = elapse / 1.5
                    radius = int(20 + progress * 40)
                    alpha = 0.6 * (1 - progress)
                    color = INK_COLORS[COLOR_IDX]
                    overlay = frame_with_ink.copy()
                    cv2.circle(overlay, (DIP_ANIMATION["x"], DIP_ANIMATION["y"]), radius, color, 2)
                    cv2.circle(overlay, (DIP_ANIMATION["x"], DIP_ANIMATION["y"]), int(radius * 0.7), color, 1)
                    frame_with_ink = cv2.addWeighted(overlay, alpha, frame_with_ink, 1 - alpha, 0)
                else:
                    DIP_ANIMATION = None

            # UI文字绘制
            cv2.putText(frame_with_ink, f"墨色:{INK_NAMES[COLOR_IDX]}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame_with_ink, f"手: {'检测到' if hand_detected else '未检测'}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            pen_text = "粗笔(拇指)" if CURRENT_WRITING_FINGER == "thumb" else "正楷(食指)"
            cv2.putText(frame_with_ink, f"笔: {pen_text}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            if BOTH_PALMS_OPEN and PALM_START_TIME:
                rem = max(0, int(PALM_DELAY - (time.time() - PALM_START_TIME)))
                status = f"换墨({rem}s)"
            else:
                status = "书写中" if is_writing else "空闲"
            cv2.putText(frame_with_ink, f"状态: {status}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame_with_ink, f"全身: {'检测到' if POSE_INFO['detected'] else '未检测'}", (10,135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            if POSE_INFO["detected"]:
                dist_txt = "近" if POSE_INFO["body_distance"] > 0.2 else "远"
                hands_txt = "合抱" if POSE_INFO["hands_together"] else "分开"
                lean_txt = "是" if POSE_INFO["lean_forward"] else "否"
                cv2.putText(frame_with_ink, f"距离:{dist_txt}", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame_with_ink, f"双手:{hands_txt}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame_with_ink, f"前倾:{lean_txt}", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.putText(frame_with_ink, "食指正楷 | 拇指粗笔 | 双手张开换墨 | 5秒自动清屏", (10,h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            cv2.putText(frame_with_ink, "按Q退出", (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("Air Calligraphy", frame_with_ink)
            cv2.imshow("Ink Canvas", ink_canvas)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n程序已退出")
    finally:
        hand_detector.close()
        pose_detector.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 初始化全局计时
    LAST_ACTIVE_TIME = time.time()
    main()
