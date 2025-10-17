#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_log(log_path: Path):
    """
    解析日志，返回两个 DataFrame
        train_df: epoch, train_loss, train_acc
        val_df  : epoch, val_loss, val_acc
    """
    train_re = re.compile(
        r'Train Epoch: (\d+).*?Average Training Metric:([\d.]+).*?Acc: ([\d.]+)'
    )
    val_re = re.compile(
        r'Validate Epoch: (\d+).*?Average Validation Metric:([\d.]+).*?Acc: ([\d.]+)'
    )

    train_records, val_records = [], []
    text = log_path.read_text(encoding='utf-8')

    for epoch, loss, acc in train_re.findall(text):
        train_records.append(
            dict(epoch=int(epoch), train_loss=float(loss), train_acc=float(acc))
        )

    for epoch, loss, acc in val_re.findall(text):
        val_records.append(
            dict(epoch=int(epoch), val_loss=float(loss), val_acc=float(acc))
        )

    train_df = pd.DataFrame(train_records).drop_duplicates('epoch').set_index('epoch')
    val_df = pd.DataFrame(val_records).drop_duplicates('epoch').set_index('epoch')

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description='Plot training/validation metrics from log.')
    parser.add_argument('--log_file', type=Path, help='Path to training log file',default='logs/v1-0-0/log2025-08-01-17-53-05.txt')
    args = parser.parse_args()

    if not args.log_file.exists():
        parser.error(f'文件不存在: {args.log_file}')

    train_df, val_df = parse_log(args.log_file)
    df = pd.concat([train_df, val_df], axis=1).sort_index()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axs[0].plot(df.index, df['train_loss'], label='Training Loss')
    axs[0].plot(df.index, df['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss Curve')
    axs[0].legend()
    axs[0].grid(True)

    # 准确率曲线
    axs[1].plot(df.index, df['train_acc'], label='Training Accuracy')
    axs[1].plot(df.index, df['val_acc'], label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy Curve')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    tgt_path = args.log_file.parent / (args.log_file.stem + '.png')
    fig.savefig(tgt_path, dpi=300)
    print(f'图像已保存到 {tgt_path}')


if __name__ == '__main__':
    main()