import os
import glob
import json
import cv2

# ========== 可调参数 ==========
VIDEO_DIR    = r"./test_videos"
SAVE_PATH    = r"./labels.json"
SPEED_FACTOR = 10       # 1=正常，2=2×快放，3=3×快放……
# ================================

def list_videos(folder: str):
    return sorted(glob.glob(os.path.join(folder, "*.mp4")))

def enlarge(frame, factor: int = 2):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)

def draw_ui(frame, text_lines, idx, total, sel_key, top_key):
    y_offset = 30
    cv2.putText(frame, f"self-involved: {sel_key}   top-down: {top_key}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    y_offset += 35
    for i, line in enumerate(text_lines):
        y = y_offset + i * 30
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 进度条
    h, w = frame.shape[:2]
    bar_left, bar_bottom = 10, h - 30
    bar_right = bar_left + 200
    bar_top   = bar_bottom - 20
    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_bottom), (0, 0, 255), -1)
    fill = bar_left + int(200 * (idx + 1) / max(total, 1))
    cv2.rectangle(frame, (bar_left, bar_top), (fill, bar_bottom), (0, 255, 0), -1)
    percent = int(100 * (idx + 1) / max(total, 1))
    cv2.putText(frame, f"{percent}%", (bar_right + 10, bar_bottom),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

def label_one_video(video_path, idx, total):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(enlarge(frame, 2))
    cap.release()
    if not frames:
        return None, None

    h, w = frames[0].shape[:2]
    win_name = f"Labeling {idx+1}/{total} – {os.path.basename(video_path)}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w, h)

    frame_idx = 0
    sel_key = top_key = None

    while True:
        frame = frames[frame_idx].copy()
        draw_ui(frame,
                ["y = self-involved, n = not, - = neither",
                 "t = top-down, b = bottom-up, = = both/neither",
                 "r = replay, <- = prev video, space/enter = next"],
                idx, total,
                sel_key if sel_key is not None else '_',
                top_key if top_key is not None else '_')

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(max(1, int(3))) & 0xFF

        if key == ord('r'):
            frame_idx = 0
            continue
        elif key in (ord('y'), ord('n'), ord('-')):
            sel_key = chr(key)
        elif key in (ord('t'), ord('b'), ord('=')):
            top_key = chr(key)
        elif key in (ord('a'), ord('q')):  # 方向键在不同系统映射不同
            print("Direction key pressed. Exiting...")
            cv2.destroyWindow(win_name)
            return "PREV", None
        elif key in (13, 32):
            if sel_key is not None and top_key is not None:
                cv2.destroyWindow(win_name)
                mapping = {'y': True, 'n': False, '-': None,
                           't': True, 'b': False, '=': None}
                return mapping[sel_key], mapping[top_key]

        frame_idx = (frame_idx + 1) % len(frames)

def main():
    videos = list_videos(VIDEO_DIR)
    if not videos:
        print("No .mp4 files found.")
        return

    labels = [None] * len(videos)
    idx = 0
    while 0 <= idx < len(videos):
        print(f"[{idx+1}/{len(videos)}] {os.path.basename(videos[idx])}")
        si, td = label_one_video(videos[idx], idx, len(videos))
        if si == "PREV":
            idx = max(0, idx - 1)
            continue
        labels[idx] = {"video": os.path.basename(videos[idx]),
                       "self_involved": si,
                       "topdown": td}
        idx += 1

    # 过滤掉 None（用户中途退出时可能出现）
    labels = [lb for lb in labels if lb is not None]
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"\nLabels saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()