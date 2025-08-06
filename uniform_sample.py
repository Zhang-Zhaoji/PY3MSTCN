#!/usr/bin/env python3
"""
一键均匀抽帧
默认：
    video_dir   = ./videos
    json_file   = ./name_feature_dict.json
    output_dir  = ./sampled_frames
如有需要，再手动传参覆盖：
    python uniform_sample.py --video_dir /my/videos --json /other.json
"""
import argparse, cv2, json, math, sys
from pathlib import Path
import numpy as np
import os
import tqdm

# ----------------- 工具函数 -----------------
def uniform_frames(video_path: Path, T: int, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"⚠ 无法读取 FPS：{video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = 0
    end_f   = total_frames - 1

    indices = np.round(np.linspace(start_f, end_f, T)).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, fid in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), frame)
    cap.release()

# ----------------- 主入口 -----------------
def main():
    parser = argparse.ArgumentParser(description="均匀抽帧脚本")
    parser.add_argument("--video_dir",  type=Path, default=Path("./sm_vids"))
    parser.add_argument("--json",       type=Path, default=Path("./data/RGBdataset/name_feature_dict.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("./clipped_sm"))
    args = parser.parse_args()

    if not args.json.exists():
        print(f"❌ JSON 文件不存在：{args.json}")
        sys.exit(1)

    name_info = json.load(open(args.json))
    stem2path = os.listdir(args.video_dir)# {p.stem: p for p in args.video_dir.iterdir() if p.is_file()}
    stem2path = [file_ for file_ in stem2path if file_.endswith(".mp4")] # select mp4 file 

    for path in tqdm.tqdm(stem2path, total= len(stem2path)):
        json_name = 'v_' + path[:-4] # remove .mp4
        data_element = name_info[json_name]
        video_path = os.path.join(args.video_dir, path)
        T = data_element['frame_count']
        out_dir = Path(os.path.join(args.output_dir, json_name))
        uniform_frames(video_path, T, out_dir)
        print(f"✅ {path} 已抽 {T} 帧 → {out_dir}")

if __name__ == "__main__":
    main()