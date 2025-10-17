import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# 1. 路径与切分下标（保持你最初的变量名）
# ==========================================================
RGB_annotation_path = 'data/annotation-Mar9th-25fps.pkl'
Sal_annotation_path = 'data/saliency_annotation.pkl'

RGB_json_path = 'test_pred/prediction_normal.json'
SM_json_path  = 'test_pred/prediction_SM.json'
MGF_json_path = 'test_pred/prediction_MGF.json'

RGB_split_num = (1355, 1355 + 290, 1355 + 290 + 290)  # 290 条
Sal_spit_num  = (1355, 1355 + 264, 1355 + 264 + 279)  # 279 条

# ==========================================================
# 2. 读 annotation & 预测结果
# ==========================================================
RGB_anno = pickle.load(open(RGB_annotation_path, 'rb'))[RGB_split_num[1]:RGB_split_num[2]]
video_names = [f'{anno[0][0]}_{int(anno[0][1])}_{int(anno[0][2])}' for anno in RGB_anno]
Sal_anno = pickle.load(open(Sal_annotation_path, 'rb'))[Sal_spit_num[1]:Sal_spit_num[2]]
print(set([element[2][0] for element in Sal_anno]))
stop_element = [element for element in Sal_anno if element[2][0] == 'Stopped']
print('stop_element:', stop_element)
print('stop_video_name:', [f'{element[0][0][2:]}_{int(element[0][1])}_{int(element[0][2])}' for element in stop_element])
quit()


RGB_pred = json.load(open(RGB_json_path))
SM_pred  = json.load(open(SM_json_path))
MGF_pred = json.load(open(MGF_json_path))

# ==========================================================
# 3. 构造 DataFrame（一行 = 一个样本的 cause 或 effect）
# ==========================================================
def build_df(anno, cause_iou, effect_iou, model_name):
    """
    返回两个 DataFrame:
        cause_df:  columns=[label, iou, model]
        effect_df: columns=[label, iou, model]
    """
    cause_rows, effect_rows = [], []
    for idx, sample in enumerate(anno):
        # sample[1] -> cause, sample[2] -> effect
        cause_label  = sample[1][0]
        effect_label = sample[2][0]

        cause_rows.append({
            'label': cause_label,
            'iou':   cause_iou[idx],
            'model': model_name
        })
        effect_rows.append({
            'label': effect_label,
            'iou':   effect_iou[idx],
            'model': model_name
        })

    return pd.DataFrame(cause_rows), pd.DataFrame(effect_rows)

# ---------------- RGB ----------------
RGB_cause_df, RGB_effect_df = build_df(
    RGB_anno,
    RGB_pred['cause_iou'],
    RGB_pred['effect_iou'],
    'RGB'
)

# ---------------- SM -----------------
SM_cause_df, SM_effect_df = build_df(
    Sal_anno,
    SM_pred['cause_iou'],
    SM_pred['effect_iou'],
    'SM'
)


# ==========================================================
# 4. 合并 & 画图
# ==========================================================
cause_df  = pd.concat([RGB_cause_df, SM_cause_df, MGF_cause_df],
                      ignore_index=True)
effect_df = pd.concat([RGB_effect_df, SM_effect_df, MGF_effect_df],
                      ignore_index=True)

sns.set_style('whitegrid')

def plot_violin(df, task='cause'):
    # 1. 计算每个 label 下 RGB 与 SM 的均值差
    mean_iou = (
        df[df['model'].isin(['RGB', 'SM'])]   # 只取 RGB 和 SM
          .groupby(['label', 'model'])['iou']
          .mean()
          .unstack(fill_value=0)
    )
    mean_iou['diff'] = mean_iou['RGB'] - mean_iou['SM']
    order = mean_iou.sort_values('diff', ascending=False).index.tolist()

    # 2. 画图
    plt.figure(figsize=(10, max(4, len(order) * 0.8)))
    sns.violinplot(
        data=df,
        x='iou',
        y='label',
        hue='model',
        order=order,       # 按均值差降序
        inner='box',
        scale='width',
        palette='Set2',
        cut=0,
        bw_adjust=0.8
    )
    plt.title(f'{task.capitalize()} IoU distribution by category\n'
              '(labels sorted by RGB − SM mean IoU, descending)')
    plt.xlabel('IoU')
    plt.ylabel('')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(f'{task}_violin_sorted.png', dpi=300)
    plt.show()

plot_violin(cause_df,  'cause')
plot_violin(effect_df, 'effect')