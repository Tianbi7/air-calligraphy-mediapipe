# paper_render.py
import cv2
import numpy as np

# 生成带噪点宣纸底色
def create_rice_paper(h, w):
    paper = np.ones((h, w, 3), dtype=np.uint8)
    paper[..., 0] = 180
    paper[..., 1] = 200
    paper[..., 2] = 230

    noise = np.random.normal(0, 4, (h, w))
    noise = np.clip(noise, -10, 10).astype(np.int8)

    paper = paper.astype(np.int16)
    paper[..., 0] = np.clip(paper[..., 0] + noise, 160, 200)
    paper[..., 1] = np.clip(paper[..., 1] + noise, 180, 220)
    paper[..., 2] = np.clip(paper[..., 2] + noise * 0.6, 210, 245)
    paper = paper.astype(np.uint8)
    return paper

# 渐变粗细毛笔墨迹渲染（楷书笔锋）
def ink_render_kai(canvas, points, brush_size, ink_color, h, w):
    if len(points) < 2:
        return canvas.copy()

    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]

        local_speed = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        base_size = max(3, int(brush_size * (0.9 - local_speed * 0.008)))

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx * dx + dy * dy)
        if length <= 0:
            continue

        steps = max(10, int(length / 2))
        start_width = base_size * 1.4
        mid_width = base_size * 0.85
        end_width = base_size * 0.3

        for step in range(steps):
            t = step / steps
            x = int(pt1[0] + dx * t)
            y = int(pt1[1] + dy * t)

            if t < 0.15:
                ratio = t / 0.15
                current_width = start_width + (mid_width - start_width) * (ratio ** 0.5)
            elif t < 0.75:
                current_width = mid_width
            else:
                ratio = (t - 0.75) / 0.25
                current_width = mid_width + (end_width - mid_width) * (ratio ** 1.5)

            sub_size = max(2, int(current_width))
            alpha = 0.6 + np.random.rand() * 0.25

            overlay = canvas.copy()
            cv2.rectangle(overlay,
                          (x - sub_size // 2, y - sub_size // 2),
                          (x + sub_size // 2, y + sub_size // 2),
                          ink_color, -1)
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

            blur_radius = max(1, sub_size // 4)
            if blur_radius % 2 == 0:
                blur_radius += 1
            y_min = max(0, y - sub_size)
            y_max = min(h, y + sub_size)
            x_min = max(0, x - sub_size)
            x_max = min(w, x + sub_size)
            blur_region = canvas[y_min:y_max, x_min:x_max]
            if blur_region.size > 0:
                canvas[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(blur_region, (blur_radius, blur_radius), sigmaX=0.5)
    return canvas
