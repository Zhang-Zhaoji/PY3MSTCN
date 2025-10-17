import numpy as np
import tqdm
import os
import json
import pickle

max_length = 0

def clip_and_save(npy_path:str, tgt_npy_path:str,ratio = 0.15,frame_count = 289):
    # 0.15 correspond t 289 frames
    npy = np.load(npy_path)
    npy_name = os.path.basename(npy_path)
    frame_c, feature_shape = npy.shape
    if ratio:
        surpressed_counts = np.round(np.linspace(0, frame_c-1, int(frame_c * ratio+0.5)))
    elif frame_count:
        surpressed_counts = np.linspace(0, frame_c-1, frame_count)
    else:
        raise ValueError("ratio or frame_count must be specified")
    npy = npy[surpressed_counts.astype(np.int32),:]
    np.save(os.path.join(tgt_npy_path, npy_name), npy)
    # global max_length
    # max_length = max(max_length, npy.shape[0])
    # print(max_length)

def main():
    full_npy_path = 'data/Saldataset/SMfeature_npy_full'
    tgt_npy_path = 'data/Saldataset/SMfeature_npy'
    full_npy_names = os.listdir(full_npy_path)
    full_npy_names = [os.path.join(full_npy_path, name) for name in full_npy_names]
    npy_metadata = pickle.load(open('data/saliency_annotation.pkl', 'rb'))
    with open('data/RGBdataset/name_feature_dict.json', 'r') as f:
        name_feature_dict = json.load(f)
    npy_names  = []
    for npy_info in tqdm.tqdm(npy_metadata):
        npy_name = npy_info[0][0]+f"_{int(npy_info[0][1])}_{int(npy_info[0][2])}"
        npy_names.append(npy_name)
    npy_names_check = set(npy_names)

    dict2list = []
    for name in name_feature_dict.keys():
        if name in npy_names_check:
            dict2list.append([name, name_feature_dict[name]])

    dict2list.sort(key=lambda x:x[1]['idx'])
    for i in range(len(dict2list)):   
        dict2list[i][1]['idx'] = i 
    
    for i, name  in enumerate(npy_names):
        # print(i, name, dict2list[i])
        assert i == dict2list[i][1]['idx'] and name == dict2list[i][0]
    print('check ok')

    for idx, name in enumerate(tqdm.tqdm(npy_names)):
        if os.path.exists(os.path.join(tgt_npy_path,  f'{idx}.npy')): continue
        npz_path = os.path.join(full_npy_path, f'{idx}.npy')
        clip_and_save(npz_path,tgt_npy_path,ratio = None,frame_count=dict2list[idx][1]['frame_count'])

if __name__ == '__main__':
    main()