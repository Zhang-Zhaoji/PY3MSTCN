import numpy as np
import tqdm
import pickle
import json
import os

if __name__ == '__main__':
    annotation_file:list[tuple[tuple[str|int]]] = pickle.load(open('data/annotation-Mar9th-25fps.pkl', 'rb'))
    with open('data/RGBdataset/name_feature_dict.json', 'r') as f:
        name_feature_dict = json.load(f)
    cnt_train = 1355
    cnt_val = 290
    cnt_test = 290
    npz_root = 'npz_sms'
    npz_files = os.listdir(npz_root)
    npz_file_names = [f.split('.')[0] for f in npz_files if f.endswith('.npz')]
    npz_file_names = set(npz_file_names)
    tgt_pt_lst = []

    NPZ_annotation = []
    output_annotation_path = 'data/saliency_annotation.pkl'
    # use set to obtain O(1) search time
    for name, feature in name_feature_dict.items():
        idx = feature['idx']
        if name not in npz_file_names:
            print(name, idx)
            if idx < 1355:
                cnt_train -= 1
            elif idx < 1355 + 290:
                cnt_val -= 1
            elif idx < 1355 + 290 + 290:
                cnt_test -= 1
        else:
            tgt_pt_lst.append([name, idx, feature['frame_count']])
    # 1355 264 279
    print(cnt_train, cnt_val, cnt_test)
    print('sort and save')
    tgt_pt_lst.sort(key=lambda x: x[1])
    print('but we need to check if the count of frames is correct')
    for i, (name, idx, frame_count) in enumerate(tqdm.tqdm(tgt_pt_lst, total=len(tgt_pt_lst))):
        # although here is not actually npz file but binary file, we can still load it
        tmp_feature = np.fromfile(os.path.join(npz_root, name + '.npz'), dtype=np.float32)
        tmp_feature = tmp_feature.reshape(-1, 512*7*7)
        assert tmp_feature.shape[0] == frame_count
        NPZ_annotation.append(annotation_file[idx])
    print('feature size is correct!')
    pickle.dump(NPZ_annotation, open(output_annotation_path, 'wb'))