import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.gridspec import GridSpec

# ==========================================================
# 1. 路径与切分下标（保持你最初的变量名）
# ==========================================================
RGB_annotation_path = 'data/annotation-Mar9th-25fps.pkl'
Sal_annotation_path = 'data/saliency_annotation.pkl'

RGB_split_num = (1355, 1355 + 290, 1355 + 290 + 290)  # 290 条
Sal_spit_num  = (1355, 1355 + 264, 1355 + 264 + 279)  # 279 条

# ==========================================================
# 2. 读 annotation & 预测结果
# ==========================================================
RGB_anno = pickle.load(open(RGB_annotation_path, 'rb'))[RGB_split_num[1]:RGB_split_num[2]]
video_names = [f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in RGB_anno]
Sal_anno = pickle.load(open(Sal_annotation_path, 'rb'))[Sal_spit_num[1]:Sal_spit_num[2]]
sal_names = set([f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in Sal_anno])

RGB_anno = [anno for anno in RGB_anno if f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' in sal_names]
for anno_RGB, anno_Sal in zip(RGB_anno, Sal_anno):
    assert anno_RGB == anno_Sal

self_saliency_dict_path = 'labels.json'
self_saliency_dict = json.load(open(self_saliency_dict_path))
dict_version = {json_dict['video']:json_dict for json_dict in self_saliency_dict}
self_saliency_dict_names = [element['video'] for element in self_saliency_dict]

acquire_labels = []
for idx, video_name in enumerate(video_names):
    name = video_name[2:]+'.mp4'
    if name in self_saliency_dict_names:
        acquire_labels.append(idx)

pred_jsonline_RGB_path = 'test_pred/RGB_predictions.jsonl'
pred_jsonline_SM_path = 'test_pred/SM_predictions.jsonl'

with open(pred_jsonline_RGB_path, 'r', encoding='utf8') as f:
    pred_jsonline_RGB = f.readlines()
pred_jsonline_RGB = [line for idx, line in enumerate(pred_jsonline_RGB) if idx in acquire_labels]
with open(pred_jsonline_SM_path, 'r', encoding='utf8') as f:
    pred_jsonline_SM = f.readlines()

pred_and_tgt = []
assert len(pred_jsonline_RGB) == len(pred_jsonline_SM)
for idx in range(len(pred_jsonline_RGB)):
    RGB_dict = json.loads(pred_jsonline_RGB[idx])
    len_RGB = int(np.sum(RGB_dict['mask']))
    SM_dict = json.loads(pred_jsonline_SM[idx])
    len_SM = int(np.sum(SM_dict['mask']))
    
    pred_and_tgt.append([RGB_dict['predicted'][:len_RGB], SM_dict['predicted'][:len_SM], RGB_dict['target'][:len_RGB], SM_dict['target'][:len_SM], np.arange(len_RGB)/len_RGB, np.arange(len_SM)/len_SM])

RGB_before_t0 = [0] * len(pred_and_tgt)
SM_before_t0 = [0] * len(pred_and_tgt)
time_diff = []
for idx, pred_and_tgt_element in enumerate(pred_and_tgt):
    time_length = RGB_anno[idx][0][2] - RGB_anno[idx][0][1]
    len_RGB = len(np.array(pred_and_tgt_element[0]))
    len_SM = len(np.array(pred_and_tgt_element[1]))
    if len(np.where(np.array(pred_and_tgt_element[0]) == 1)) > 0 and len(np.where(np.array(pred_and_tgt_element[1]) == 1)) > 0:
        pass
    else:
        continue
    try:
        first_t0_RGB = np.where(np.array(pred_and_tgt_element[0]) == 1)[0][0]
    except:
        first_t0_RGB = np.nan
    try:
        first_t0_SM = np.where(np.array(pred_and_tgt_element[1]) == 1)[0][0]
    except:
        first_t0_SM = np.nan
    first_t0_RGB = first_t0_RGB / len_RGB
    first_t0_SM = first_t0_SM / len_SM
    try:
        first_t0_GT_RGB = np.where(np.array(pred_and_tgt_element[2]) == 1)[0][0]
    except:
        first_t0_GT_RGB = 0
    first_t0_GT_RGB = first_t0_GT_RGB / len_RGB
    try:
        first_t0_GT_SM = np.where(np.array(pred_and_tgt_element[3]) == 1)[0][0]
    except:
        first_t0_GT_SM = 0
    first_t0_GT_SM = first_t0_GT_SM / len_SM
    RGB_before_t0[idx] = (first_t0_GT_RGB - first_t0_RGB) * time_length if not np.isnan(first_t0_RGB) else 0
    SM_before_t0[idx] = (first_t0_GT_SM - first_t0_SM) * time_length if not np.isnan(first_t0_SM) else 0
    time_diff.append(first_t0_GT_RGB * time_length - first_t0_GT_SM * time_length)

x_mean = np.mean(RGB_before_t0)
x_std = np.std(RGB_before_t0)
y_mean = np.mean(SM_before_t0)
y_std = np.std(SM_before_t0)

RGB_before_t0 = np.array(RGB_before_t0)
SM_before_t0 = np.array(SM_before_t0)

RGB_non0 = RGB_before_t0[RGB_before_t0 != 0]
SM_non0 = SM_before_t0[SM_before_t0 != 0]

t_rgb, p_rgb = stats.ttest_1samp(RGB_non0, 0)
t_sm, p_sm = stats.ttest_1samp(SM_non0, 0)

print(f'RGB  均值={RGB_non0.mean():.3f},  std={RGB_non0.std():.3f}')
print(f'     t={t_rgb:6.3f},  p={p_rgb:.4g}  (df={len(RGB_non0)-1})')
print(f'SM   均值={SM_non0.mean():.3f},  std={SM_non0.std():.3f}')
print(f'     t={t_sm:6.3f},  p={p_sm:.4g}  (df={len(SM_non0)-1})')

df = pd.DataFrame({'RGB_before_t0': RGB_before_t0, 'SM_before_t0': SM_before_t0})

# ========== 关键修改：创建复杂布局 ==========

fig = plt.figure(figsize=(8, 8))

# 使用 GridSpec 创建布局
gs = GridSpec(4, 4,  height_ratios=[1, 1, 7, 1], width_ratios=[1, 7, 1, 1], wspace=0.1, hspace=0.1)

# 主图：核心区域 [-4, 4] x [-4, 4]
ax_main = fig.add_subplot(gs[2, 1])
sns.scatterplot(data=df, x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_main, alpha=0.7)
ax_main.set_aspect('equal')

# 设置主图范围
ax_main.set_xlim(-4, 4)
ax_main.set_ylim(-4, 4)
ax_main.set_xlabel('')
ax_main.set_ylabel('')
# ax_main.set_xlabel('RGB prediction lead time (s)', fontsize=12)
# ax_main.set_ylabel('SM prediction lead time (s)', fontsize=12)
ax_main.axvline(x_mean, color='r', lw=1, ls='--')
ax_main.axhline(y_mean, color='r', lw=1, ls='--')
ax_main.axvline(0, color='gray', lw=1, ls='--')
ax_main.axhline(0, color='gray', lw=1, ls='--')
ax_main.tick_params(labelbottom=False, labelleft=False)
# ax_main.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='none', capsize=5, color='r', lw=3)

# 添加统计文本到主图
def p2star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

star_rgb = p2star(p_rgb)
star_sm = p2star(p_sm)
text_info = (f'mean: ({x_mean:.2f}, {y_mean:.2f})\n'
             f'std: ({x_std:.2f}, {y_std:.2f})\n'
             f'RGB: t={t_rgb:5.2f}, p={p_rgb:.2g} {star_rgb}\n'
             f'SM : t={t_sm:5.2f}, p={p_sm:.2g} {star_sm}')

ax_main.text(0.05, 0.25, text_info,
             transform=ax_main.transAxes,
             va='top', ha='left', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

ax_marg_x = fig.add_subplot(gs[0, 1], sharex=ax_main)
sns.histplot(data=df, x='RGB_before_t0', bins=40, kde=True, ax=ax_marg_x)
ax_marg_x.set_xlim(-4, 4)
ax_marg_x.tick_params(labelbottom=False, labelleft=True)
ax_marg_x.set_ylabel('Count', fontsize=10)
ax_marg_x.set_xlabel('', fontsize=10)

# 右侧边缘图 (Y分布)
ax_marg_y = fig.add_subplot(gs[2, 3], sharey=ax_main)
sns.histplot(data=df, y='SM_before_t0', bins=40, kde=True, ax=ax_marg_y)
ax_marg_y.set_ylim(-4, 4)
ax_marg_y.tick_params(labelleft=False, labelbottom=True)
ax_marg_y.set_xlabel('Count', fontsize=10)
ax_marg_y.set_ylabel('', fontsize=10)

# 左侧小图：
ax_left = fig.add_subplot(gs[2, 0], sharey=ax_main)
sns.scatterplot(data=df[(df['RGB_before_t0'] <= 4) & (df['SM_before_t0'] <= 4) & (df['SM_before_t0'] >= -4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_left)
ax_left.set_xlim(-15, -4)
ax_left.set_ylim(-4, 4)
ax_left.set_xlabel('')
ax_left.set_ylabel('')
ax_left.axhline(y_mean, color='r', lw=1, ls='--')
ax_left.axhline(0, color='gray', lw=1, ls='--')
ax_left.tick_params(labelbottom=False, labelleft=True)

# 顶部小图：
ax_top = fig.add_subplot(gs[1, 1], sharex=ax_main)
sns.scatterplot(data=df[(df['RGB_before_t0'] >= -4) &(df['RGB_before_t0'] <= 4) & (df['SM_before_t0'] >= 4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_top)
ax_top.set_xlim(-4, 4)
ax_top.set_ylim(4, 15)
ax_top.set_xlabel('')
ax_top.set_ylabel('')
ax_top.axvline(x_mean, color='r', lw=1, ls='--')
ax_top.axvline(0, color='gray', lw=1, ls='--')
ax_top.tick_params(labelbottom=False, labelleft=False)

# 右侧小图：
ax_right = fig.add_subplot(gs[2, 2], sharey=ax_main)
sns.scatterplot(data=df[(df['RGB_before_t0'] >= 4) & (df['SM_before_t0'] <= 4) & (df['SM_before_t0'] >= -4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_right)
ax_right.set_xlim(4, 15)
ax_right.set_ylim(-4, 4)
ax_right.set_xlabel('')
ax_right.set_ylabel('')
ax_right.axhline(y_mean, color='r', lw=1, ls='--')
ax_right.axhline(0, color='gray', lw=1, ls='--')
ax_right.tick_params(labelbottom=False, labelleft=False)

# 底部小图：
ax_bottom = fig.add_subplot(gs[3, 1], sharex=ax_main)
sns.scatterplot(data=df[(df['RGB_before_t0'] >= -4) & (df['RGB_before_t0'] <= 4) & (df['SM_before_t0'] <= -4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_bottom)
ax_bottom.set_xlim(-4, 4)
ax_bottom.set_ylim(-15, -4)
ax_bottom.set_xlabel('')
ax_bottom.set_ylabel('')
ax_bottom.axvline(x_mean, color='r', lw=1, ls='--')
ax_bottom.axvline(0, color='gray', lw=1, ls='--')
ax_bottom.tick_params(labelbottom=True, labelleft=False)


# 左下小图：
ax_bottom_left = fig.add_subplot(gs[3, 0], sharex=ax_left, sharey=ax_bottom)
sns.scatterplot(data=df[(df['RGB_before_t0'] <= -4) & (df['RGB_before_t0'] <= -4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_bottom_left)
ax_bottom_left.set_xlim(-15, -4)
ax_bottom_left.set_ylim(-15, -4)
ax_bottom_left.set_xlabel('')
ax_bottom_left.set_ylabel('')
ax_bottom_left.tick_params(labelbottom=True, labelleft=True)

# 左上小图：
ax_top_left = fig.add_subplot(gs[1, 0], sharex=ax_left, sharey=ax_top)
sns.scatterplot(data=df[(df['RGB_before_t0'] <= -4) & (df['SM_before_t0'] >= 4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_top_left)
ax_top_left.set_xlim(-15, -4)
ax_top_left.set_ylim(4, 15)
ax_top_left.set_xlabel('')
ax_top_left.set_ylabel('')
ax_top_left.tick_params(labelbottom=False, labelleft=True)

# 右上小图：
ax_top_right = fig.add_subplot(gs[1, 2], sharex=ax_right, sharey=ax_top)
sns.scatterplot(data=df[(df['RGB_before_t0'] >= 4)&(df['SM_before_t0'] >= 4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_top_right)
ax_top_right.set_xlim(4, 15)
ax_top_right.set_ylim(4, 15)
ax_top_right.set_xlabel('')
ax_top_right.set_ylabel('')
ax_top_right.tick_params(labelbottom=False, labelleft=False)

# 右下小图：
ax_bottom_right = fig.add_subplot(gs[3, 2], sharex=ax_right, sharey=ax_bottom)
sns.scatterplot(data=df[(df['RGB_before_t0'] >= 4) & (df['SM_before_t0'] <= -4)],
                x='RGB_before_t0', y='SM_before_t0', s=20, color='blue', ax=ax_bottom_right)
ax_bottom_right.set_xlim(4, 15)
ax_bottom_right.set_ylim(-15, -4)
ax_bottom_right.set_xlabel('')
ax_bottom_right.set_ylabel('')
ax_bottom_right.tick_params(labelbottom=True, labelleft=False)

ax_bottom.set_xlabel('RGB prediction lead time (s)', fontsize=18)
# 将Y轴标签放在左侧小图上
ax_left.set_ylabel('SM prediction lead time (s)', fontsize=18)

plt.tight_layout()
plt.show()