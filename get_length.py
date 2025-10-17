#!/usr/bin/env python3
"""
遍历指定目录下所有视频文件，输出最高帧数。
用法：
    python get_max_frame_count.py /path/to/dir
"""
import os
import sys
import cv2

# 常见视频扩展名（按需增删）
VIDEO_EXTS = {
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.mts', '.ts'
}

def get_frame_count(video_path):
    """返回视频总帧数；若失败返回 None。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    # 优先用 CAP_PROP_FRAME_COUNT；若失败再手动 seek
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 某些文件元数据不准，尝试手动 seek 到结尾
    if frames <= 0:
        frames = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frames += 1
    cap.release()
    return frames if frames > 0 else None

def main(root_dir):
    if not os.path.isdir(root_dir):
        print(f"路径不存在: {root_dir}")
        sys.exit(1)

    max_frames = 0
    max_file = ""

    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTS:
                full_path = os.path.join(dirpath, f)
                frames = get_frame_count(full_path)
                if frames is None:
                    print(f"[跳过] 无法读取: {full_path}")
                    continue
                print(f"{frames:>10} 帧  {full_path}")
                if frames > max_frames:
                    max_frames = frames
                    max_file = full_path

    if max_frames == 0:
        print("未找到任何可识别的视频文件。")
    else:
        print("\n" + "="*60)
        print(f"最高帧数: {max_frames}")
        print(f"文件路径: {max_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python get_max_frame_count.py <文件夹路径>")
        sys.exit(1)
    main(sys.argv[1])