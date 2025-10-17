import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import jsonlines
from scipy import stats

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
    # len_RGB = len(np.sum(RGB_dict['mask']))
    len_RGB = int(np.sum(RGB_dict['mask']))
    SM_dict = json.loads(pred_jsonline_SM[idx])
    #len_SM = len(SM_dict['mask'])
    len_SM = int(np.sum(SM_dict['mask']))
    
    # print(RGB_dict['predicted'][0], SM_dict['predicted'][0])
    pred_and_tgt.append([RGB_dict['predicted'][:len_RGB], SM_dict['predicted'][:len_SM], RGB_dict['target'][:len_RGB], SM_dict['target'][:len_SM], np.arange(len_RGB)/len_RGB, np.arange(len_SM)/len_SM])

# for i in range(len(pred_and_tgt)):
#     plt.plot(pred_and_tgt[i][5], pred_and_tgt[i][1], label='SM')
#     plt.plot(pred_and_tgt[i][5], pred_and_tgt[i][3], label='SM_GT')
#     plt.plot(pred_and_tgt[i][4], pred_and_tgt[i][0], label='RGB')
#     plt.plot(pred_and_tgt[i][4], pred_and_tgt[i][2], label='RGB_GT')
#     plt.legend()
#     plt.show()

RGB_before_t0 = [0] * len(pred_and_tgt)
SM_before_t0 = [0] * len(pred_and_tgt)
time_diff = []
for idx, pred_and_tgt_element in enumerate(pred_and_tgt):
    time_length = RGB_anno[idx][0][2] - RGB_anno[idx][0][1]
    len_RGB =  len(np.array(pred_and_tgt_element[0]))
    len_SM = len(np.array(pred_and_tgt_element[1]))
    print('idx = ',idx)
    if len(np.where(np.array(pred_and_tgt_element[0]) == 1)) > 0 and len(np.where(np.array(pred_and_tgt_element[1]) == 1)) > 0:
        pass
        print(len(np.where(np.array(pred_and_tgt_element[0]) == 1)), len(np.where(np.array(pred_and_tgt_element[1]) == 1)))
    else:
        continue
    # print(pred_and_tgt_element)
    try:
        first_t0_RGB = np.where(np.array(pred_and_tgt_element[0]) == 1)[0][0]
    except:
        first_t0_RGB = np.nan
    print(first_t0_RGB)
    try:
        first_t0_SM = np.where(np.array(pred_and_tgt_element[1]) == 1)[0][0]
    except:
        first_t0_SM = np.nan
    print(first_t0_SM)
    first_t0_RGB = first_t0_RGB/ len_RGB
    first_t0_SM = first_t0_SM/ len_SM
    try:
        first_t0_GT_RGB = np.where(np.array(pred_and_tgt_element[2]) == 1)[0][0]
    except:
        first_t0_GT_RGB=0
    
    print(first_t0_GT_RGB)
    first_t0_GT_RGB = first_t0_GT_RGB / len_RGB

    try:
        first_t0_GT_SM = np.where(np.array(pred_and_tgt_element[3]) == 1)[0][0]
    except:
        first_t0_GT_SM=0
    
    print(first_t0_GT_SM)
    first_t0_GT_SM = first_t0_GT_SM/ len_SM
    RGB_before_t0[idx] = (first_t0_GT_RGB - first_t0_RGB)*time_length if not np.isnan(first_t0_RGB) else 0
    SM_before_t0[idx] = (first_t0_GT_SM - first_t0_SM)*time_length if not np.isnan(first_t0_SM) else 0
    time_diff.append(first_t0_GT_RGB*time_length - first_t0_GT_SM*time_length)

x_mean = np.mean(RGB_before_t0)
x_std = np.std(RGB_before_t0)
y_mean = np.mean(SM_before_t0)
y_std = np.std(SM_before_t0)



# 你已经有的向量
RGB_before_t0 = np.array(RGB_before_t0)
SM_before_t0  = np.array(SM_before_t0)

# 去掉 0 值（如果前面想排除未成功样本）
RGB_non0 = RGB_before_t0[RGB_before_t0 != 0]
SM_non0  = SM_before_t0[SM_before_t0 != 0]

# ---- 检验 ----
t_rgb, p_rgb = stats.ttest_1samp(RGB_non0, 0)
t_sm,  p_sm  = stats.ttest_1samp(SM_non0,  0)

print(f'RGB  均值={RGB_non0.mean():.3f},  std={RGB_non0.std():.3f}')
print(f'     t={t_rgb:6.3f},  p={p_rgb:.4g}  (df={len(RGB_non0)-1})')

print(f'SM   均值={SM_non0.mean():.3f},  std={SM_non0.std():.3f}')
print(f'     t={t_sm:6.3f},  p={p_sm:.4g}  (df={len(SM_non0)-1})')


# 先把数据整理成 DataFrame，方便 seaborn 调用
df = pd.DataFrame({'RGB_before_t0': RGB_before_t0,
                   'SM_before_t0':  SM_before_t0})

# 1. 一键出图
sns.set_context({'figure.figsize':[6, 6]})
g = sns.jointplot(data=df, x='RGB_before_t0', y='SM_before_t0', s=20, marginal_kws=dict(bins=20, fill=False),color='black')
g.ax_joint.set_xlabel('RGB prediction lead time (s)',fontdict={'size': 16})
g.ax_joint.set_ylabel('SM prediction lead time (s)',fontdict={'size': 16})
g.ax_joint.set_aspect('equal', adjustable='box')

g.ax_joint.axvline(x_mean, color='r', lw=1, ls='--')
g.ax_joint.axhline(y_mean, color='r', lw=1, ls='--')

g.ax_joint.errorbar(x_mean, y_mean,
                    xerr=x_std, yerr=y_std,
                    fmt='none',
                    capsize=5,
                    color='r', lw=3)

# ---------- 星号映射 ----------
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
star_sm  = p2star(p_sm)

# ---------- 组装文字 ----------
text_info = (f'mean: ({x_mean:.2f}, {y_mean:.2f})\n'
             f'std: ({x_std:.2f}, {y_std:.2f})\n'
             f'RGB: t={t_rgb:5.2f}, p={p_rgb:.2g} {star_rgb}\n'
             f'SM : t={t_sm:5.2f}, p={p_sm:.2g} {star_sm}')


g.ax_joint.text(0.5, 1.0, text_info,
                transform=g.ax_joint.transAxes,
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.show()

