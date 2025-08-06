#!/usr/bin/env python3
"""
一次性遍历 CTA 数据集里所有 .pt 文件，打印张量维度。
用法：
    python inspect_cta_pt.py /path/to/CTA/dataset
"""
import os
import sys
import torch
from pathlib import Path
import numpy as np
import pickle
from typing import Any, Dict, List, Tuple, Union
import json

Tensor = torch.Tensor

def load_pt(path: Union[str, Path]) -> List[Tensor]:
    """
    Load a .pt file.
    """
    return torch.load(path, map_location="cpu")

def load_annotation(path: Union[str, Path]) -> List[Tuple[str, Any]]:
    """
    Load a .pkl file.
    """
    return pickle.load(open(path, "rb"))

def comparision() -> None:
    feats = load_pt('./i3d-rgb-flip-fps25-Mar9th.pt')
    annos = load_annotation('../annotation-Mar9th-25fps.pkl')

    name_feature_dict = {}

    if len(feats) != len(annos):
        print("⚠  特征与标注数量不一致，请检查文件！")
        return

    for idx, (feat, (anno, *_)) in enumerate(zip(feats, annos)):
        video_name, start_time, end_time, *_rest = anno
        delta_t = end_time - start_time
        frames_25fps = delta_t * 25
        T = feat.shape[0]
        ratio = frames_25fps / T
        print(
            f"{idx:4d} | {video_name:<20s} "
            f"| {start_time:6.2f}–{end_time:6.2f} s "
            f"| Δt={delta_t:6.2f} s "
            f"| 25fps={frames_25fps:6.1f} "
            f"| T={T:3d} "
            f"| ratio={ratio:.2f}"
        )
        name_feature_dict[f'{video_name}_{int(start_time)}_{int(end_time)}'] = {'idx': idx, 'frame_count': T}
    json.dump(name_feature_dict, open('name_feature_dict.json', 'w'),indent=4)


def main() -> None:
    """
    Main function.
    """
    # path = sys.argv[1]
    comparision()

if __name__ == "__main__":
    main()

