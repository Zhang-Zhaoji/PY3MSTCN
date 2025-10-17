import os
import re
import json
import ast
from glob import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_patterns()->list[re.Pattern]:
    """
    generate some patterns to detect some information in the log file.

    return a list of patterns.
    1. parameter detector: detect dict-like parameters,should be like:
    { "model_name": "v2-1-32", "save_model": true, "result_path": "./rsts/MGFv2-1-32", "max_save_model": 3, "lr": 0.001, "lr_decay": 0.95, "lr_decay_step": 15, "weight_decay": 0.01, "optimizer": "AdamW", "scheduler": "StepLR", "resume": "", "num_classes": 3, "num_stages": 4, "num_layers": 10, "num_f_maps": 32, "dim": 1024, "epochs": 100 }
    2. best epoch number detector: detect best epoch number, should be like 
    [2025-08-09 17:43:42] [INFO] Best epoch: 7 | Best Combined IoU@0.5: 0.1155
    3.  best epoch's iou information detector: detect best epoch's iou information, should be like: 
        [2025-08-09 17:43:01] [INFO] Validate Epoch: 97, Average Validation Metric:1.410863 = 372.467909/264.000000, Acc: 0.7283, Cause IoU: 0.1285, Effect IoU: 0.1479]	
        [2025-08-09 17:43:01] [INFO] Validation Key IoU Thresholds:
        [2025-08-09 17:43:01] [INFO] Cause IoU:
        [2025-08-09 17:43:01] [INFO]   IoU > 0.1: 0.4167
        [2025-08-09 17:43:01] [INFO]   IoU > 0.3: 0.1742
        [2025-08-09 17:43:01] [INFO]   IoU > 0.5: 0.0379
        [2025-08-09 17:43:01] [INFO]   IoU > 0.7: 0.0076
        [2025-08-09 17:43:01] [INFO] Effect IoU:
        [2025-08-09 17:43:01] [INFO]   IoU > 0.1: 0.3636
        [2025-08-09 17:43:01] [INFO]   IoU > 0.3: 0.2197
        [2025-08-09 17:43:01] [INFO]   IoU > 0.5: 0.0947
        [2025-08-09 17:43:01] [INFO]   IoU > 0.7: 0.0379
        [2025-08-09 17:43:01] [INFO] Combined IoU:
        [2025-08-09 17:43:01] [INFO]   IoU > 0.1: 0.3902
        [2025-08-09 17:43:01] [INFO]   IoU > 0.3: 0.1970
        [2025-08-09 17:43:01] [INFO]   IoU > 0.5: 0.0663
        [2025-08-09 17:43:01] [INFO]   IoU > 0.7: 0.0227
        [2025-08-09 17:43:01] [INFO] Validation ends: 
    4. best training iou information detector: detect best training iou information, should be like:
        [2025-08-09 17:43:17] [INFO] Learning rate: 0.001
        [2025-08-09 17:43:17] [INFO] Cause  0.1 ~ 0.9 iou: tensor([0.9993, 0.9985, 0.9978, 0.9970, 0.9963, 0.9948, 0.9838, 0.9697, 0.9255])
        [2025-08-09 17:43:17] [INFO] Effect 0.1 ~ 0.9 iou: tensor([0.9860, 0.9860, 0.9845, 0.9845, 0.9808, 0.9786, 0.9720, 0.9609, 0.9439])
        [2025-08-09 17:43:17] [INFO] Both   0.1 ~ 0.9 iou: tensor([0.9926, 0.9923, 0.9911, 0.9908, 0.9886, 0.9867, 0.9779, 0.9653, 0.9347])
        [2025-08-09 17:43:17] [INFO] Key IoU Thresholds:
        [2025-08-09 17:43:17] [INFO] Cause IoU:
        [2025-08-09 17:43:17] [INFO]   IoU > 0.1: 0.9993
        [2025-08-09 17:43:17] [INFO]   IoU > 0.3: 0.9978
        [2025-08-09 17:43:17] [INFO]   IoU > 0.5: 0.9963
        [2025-08-09 17:43:17] [INFO]   IoU > 0.9: 0.9255
        [2025-08-09 17:43:17] [INFO] Effect IoU:
        [2025-08-09 17:43:17] [INFO]   IoU > 0.1: 0.9860
        [2025-08-09 17:43:17] [INFO]   IoU > 0.3: 0.9845
        [2025-08-09 17:43:17] [INFO]   IoU > 0.5: 0.9808
        [2025-08-09 17:43:17] [INFO]   IoU > 0.9: 0.9439
        [2025-08-09 17:43:17] [INFO] Epoch time: 15.90s
        [2025-08-09 17:43:17] [INFO] Training ends: 
    """
    patterns = []
    
    # 1. Parameter detector
    param_pattern = re.compile(r'\{\s*"[^}]+\}')
    patterns.append(param_pattern)
    
    # 2. Best epoch number detector
    best_epoch_pattern = re.compile(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Best epoch: (\d+) \| Best Combined IoU@0\.5: \d+\.\d+')
    patterns.append(best_epoch_pattern)
    
    # 3. Best epoch's validation IoU information detector
    validation_iou_pattern = re.compile(r"""
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Validate Epoch: \d+, Average Validation Metric:\d+\.\d+ = \d+\.\d+/\d+\.\d+, Acc: \d+\.\d+, Cause IoU: \d+\.\d+, Effect IoU: \d+\.\d+[\]]* 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Validation Key IoU Thresholds: 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Cause IoU: 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.1: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.3: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.5: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.7: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Effect IoU: 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.1: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.3: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.5: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.7: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Combined IoU: 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.1: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.3: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.5: (\d+\.\d+) 
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\]   IoU > 0.7: (\d+\.\d+) 
[\s\S]*?
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Validation ends:""".replace("\n", ""))
    patterns.append(validation_iou_pattern)
    # 4. Best training IoU information detector - 
    training_iou_pattern = re.compile(r"""\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Train Epoch: \d+, Average Training Metric:\d+\.\d+ = \d+\.\d+/\d+\.\d+, Acc: \d+\.\d+, Cause IoU: \d+\.\d+, Effect IoU: \d+\.\d+\]\s*?
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Learning rate: [\S\s]*?
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Cause  0\.1 ~ 0\.9 iou: tensor\(\[(\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, \d+\.\d*?\]\)\s*?
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Effect 0\.1 ~ 0\.9 iou: tensor\(\[(\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, \d+\.\d*?\]\)\s*?
\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[INFO\] Both   0\.1 ~ 0\.9 iou: tensor\(\[(\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, (\d+\.\d*?), \d+\.\d*?, \d+\.\d*?\]\)""".replace("\n", ""))
    patterns.append(training_iou_pattern)
    
    return patterns

def read_log(log_path:str, patterns:list[re.Pattern])->list[str,int,list[float],list[float]|None]:
    """
    pre_write some patterns, try to find corresponding records in the log file.

    if exists, return a list of strings of the records.

    if not exists, return an empty list.

    ===========================================

    patterns should contain:
        1.parameter detector: detect dict-like parameters\{\s*\};
        2.best epoch number detector: detect best epoch number, should be like [2025-08-09 17:43:42] [INFO] Best epoch: 7 | Best Combined IoU@0.5: 0.1155;
        3.best epoch's iou information detector: detect best epoch's iou information, should be like:
        4. best training iou information detector: detect best training iou information, should be like:

    ===========================================

    returned list should contain:

    1.the parameterline;

    2.best epoch number;

    3.best epoch's iou information(validation set);

    4.best training iou information;

    """
    print(f'reading log file: {log_path}')
    if not os.path.exists(log_path): return []
    log_file = open(log_path, "r").readlines()
    log_file = [line.strip() for line in log_file]
    log_data = ' '.join(log_file)
    del log_file
    parameters = re.findall(patterns[0], log_data)
    if len(parameters) == 0: return []
    parameters = parameters[0]
    best_epoch_number = re.findall(patterns[1], log_data)
    if len(best_epoch_number) == 0:
    #     print('no best epoch number found, the log file may be incomplete or too old.')
        return []
    best_epoch_number = int(best_epoch_number[-1])
    print(f'best epoch number: {best_epoch_number}')

    validation_iou = re.findall(patterns[2], log_data)
    if len(validation_iou) == 0: return []
    # print(f'found {len(validation_iou)} validation iou data elements in the log file.')
    best_validation_ious = validation_iou[best_epoch_number]
    # print('best validation iou: ')
    #rst_info_strs = ['best epoch:', 'Cause IoU>0.1:','Cause IoU>0.3:','Cause IoU>0.5:','Cause IoU>0.7:', 'Effect IoU>0.1:', 'Effect IoU>0.3:', 'Effect IoU>0.5:', 'Effect IoU>0.7:', 'Combined IoU>0.1:', 'Combined IoU>0.3:', 'Combined IoU>0.5:', 'Combined IoU>0.7:', ]
    #for (rst_info_str, best_val_iou) in zip(rst_info_strs, best_validation_ious):
    #    print(f'{rst_info_str} {best_val_iou}')
    
    training_iou = re.findall(patterns[3], log_data)
    if len(training_iou) == 0: return []
    print(f'found {len(training_iou)} training iou data elements in the log file.')
    try:
        best_training_ious = training_iou[best_epoch_number]
    except IndexError:
        print('best epoch number is out of range.')
        print(training_iou)
        exit(1)
    #print('best training iou: ')
    #rst_info_strs = ['best epoch:', 'Cause IoU>0.1:','Cause IoU>0.3:','Cause IoU>0.5:','Cause IoU>0.7:', 'Effect IoU>0.1:', 'Effect IoU>0.3:', 'Effect IoU>0.5:', 'Effect IoU>0.7:', 'Combined IoU>0.1:', 'Combined IoU>0.3:', 'Combined IoU>0.5:', 'Combined IoU>0.7:', ]
    #for (rst_info_str, best_val_iou) in zip(rst_info_strs, best_training_ious):
    #    print(f'{rst_info_str} {best_val_iou}')
    return [parameters, best_epoch_number, best_validation_ious, best_training_ious]

def logdata_to_df(log_data_list: List[List[Any]]) -> pd.DataFrame:
    """
    将 read_log 返回的 log_data_list 拉平成一张 DataFrame
    列：
        model_name, lr, num_stages, num_layers, num_f_maps, ...
        best_epoch
        val_cause_0.1, val_cause_0.3, ..., val_combined_0.7
        train_cause_0.1, ..., train_both_0.9
    """
    rows = []
    for params_str, best_epoch, val_ious, train_ious in log_data_list:
        # 1. 解析参数字典
        # 用 ast.literal_eval 比 json.loads 容错更好（单引号/ true -> True）
        params_str = re.sub(r'\btrue\b', 'True', params_str)
        params_str = re.sub(r'\bfalse\b', 'False', params_str)
        params: Dict[str, Any] = ast.literal_eval(params_str)

        # 2. 解析 IoU
        # val_ious 长度 12，顺序：
        # Cause>0.1,0.3,0.5,0.7
        # Effect>0.1,0.3,0.5,0.7
        # Combined>0.1,0.3,0.5,0.7
        val_keys = [
            "val_cause_0.1", "val_cause_0.3", "val_cause_0.5", "val_cause_0.7",
            "val_effect_0.1", "val_effect_0.3", "val_effect_0.5", "val_effect_0.7",
            "val_combined_0.1", "val_combined_0.3", "val_combined_0.5", "val_combined_0.7"
        ]
        # train_ious 长度 9，顺序：
        # Cause 0.1,0.3,0.5  （我们只取 0.1,0.3,0.5）
        # Effect 0.1,0.3,0.5
        # Both   0.1,0.3,0.5
        train_keys = [
            "train_cause_0.1", "train_cause_0.3", "train_cause_0.5", "train_cause_0.7",
            "train_effect_0.1", "train_effect_0.3", "train_effect_0.5", "train_effect_0.7",
            "train_both_0.1", "train_both_0.3", "train_both_0.5", "train_both_0.7",
        ]

        row = {**params, "best_epoch": best_epoch}
        row.update(dict(zip(val_keys, map(float, val_ious))))
        row.update(dict(zip(train_keys, map(float, train_ious))))
        rows.append(row)

    df = pd.DataFrame(rows)
    # 把 bool -> int，方便分组
    df["save_model"] = df["save_model"].astype(int)
    return df


# ------------------------------------------------
# 2. 主流程：读取 -> 建表 -> 画图
# ------------------------------------------------
def main():
    patterns = generate_patterns()
    log_root = "logs"
    log_files = glob(os.path.join(log_root, "*/*.txt"))

    log_data_list = []
    for log_file in tqdm(log_files, desc="reading logs"):
        data = read_log(log_file, patterns)
        if data:
            log_data_list.append(data)

    df = logdata_to_df(log_data_list)
    print("共解析出实验数:", len(df))
    print(df.head())

    # ------------------------------------------------
    # 3. 画图
    # ------------------------------------------------
    sns.set_theme(style="whitegrid")

    # 3.1 不同 num_f_maps 下的 val combined IoU@0.5
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="num_f_maps", y="val_combined_0.5")
    plt.title("Validation Combined IoU@0.5 vs num_f_maps")
    plt.tight_layout()
    plt.show()

    # 3.2 不同学习率 lr vs val_combined_0.5
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="lr", y="val_combined_0.5", hue="num_layers", palette="viridis")
    plt.xscale("log")
    plt.title("Val Combined IoU@0.5 vs lr (color=num_layers)")
    plt.tight_layout()
    plt.show()

    # 3.3 训练/验证曲线对比（热力图）
    # 以 num_stages 为 x，num_layers 为 y，val_combined_0.5 为值
    pivot = df.pivot_table(values="val_combined_0.5",
                           index="num_layers",
                           columns="num_stages",
                           aggfunc="mean")
    plt.figure(figsize=(5, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Val Combined IoU@0.5 heatmap")
    plt.tight_layout()
    plt.show()

    # 3.4 同一实验 train vs val 对比（对角线）
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=df["train_both_0.5"], y=df["val_combined_0.5"])
    plt.plot([0, 1], [0, 1], ls="--", c="grey")
    plt.xlabel("Train Both IoU@0.5")
    plt.ylabel("Val Combined IoU@0.5")
    plt.title("Train vs Val")
    plt.tight_layout()
    plt.show()

    # 假设 df 已经准备好
    row_label = (df['num_layers'].astype(str) + '-' +
                 df['num_stages'].astype(str) + '-' +
                 df['num_f_maps'].astype(str) + '-' +
                 df['dim'].astype(str))

    # 把新行标签挂到 DataFrame 上
    tmp = df.assign(row_label=row_label)

    # 生成普通二维透视表
    pivot = (tmp
             .pivot_table(values='val_combined_0.5',
                          index='row_label',
                          columns='model_name',
                          aggfunc='max')
             .sort_index())   # 行按字典序排

    # 打印
    print('\n=== 超参数组合 vs model_name 的 val Combined IoU@0.5 ===')
    print(pivot.round(4))

    # 导出
    pivot.to_excel('param_combo_vs_model.xlsx')

if __name__ == '__main__':
    main()

"""
=== 超参数组合 vs model_name 的 val Combined IoU@0.5 ===
model_name     v2-0-0  v2-1-128  v2-1-32  v2-2-128  v2-2-256  v2-2-32
row_label
10-4-128-1024  0.5914    0.3295      NaN    0.4640    0.4356      NaN
10-4-128-512      NaN       NaN      NaN    0.4413       NaN      NaN
10-4-32-1024   0.5483       NaN   0.3201       NaN       NaN   0.3996
12-4-128-1024     NaN       NaN      NaN    0.4394       NaN      NaN
12-4-128-2048     NaN       NaN      NaN    0.4223       NaN      NaN
12-4-144-1024     NaN       NaN      NaN    0.4337       NaN      NaN
12-4-196-1024     NaN       NaN      NaN    0.4129       NaN      NaN
12-4-96-1024      NaN       NaN      NaN    0.4205       NaN      NaN
12-5-128-1024     NaN       NaN      NaN    0.0000       NaN      NaN
13-4-128-1024     NaN       NaN      NaN    0.4432       NaN      NaN
15-4-128-1024     NaN       NaN      NaN    0.4337       NaN      NaN
2-2-32-1024    0.3621       NaN      NaN       NaN       NaN      NaN
3-4-128-1024   0.4448       NaN      NaN       NaN       NaN      NaN
5-4-128-1024      NaN       NaN      NaN    0.4167       NaN      NaN
8-4-128-1024      NaN       NaN      NaN    0.4413       NaN      NaN
"""