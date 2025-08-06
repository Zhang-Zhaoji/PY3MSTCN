# convert_pt_to_npy_individual.py
import torch
import numpy as np
import os

pt_path = 'data/Saldataset/MGFfeature.pt'
save_dir = 'data/Saldataset/MGFfeature_npy'

os.makedirs(save_dir, exist_ok=True)
print(f"Loading {pt_path}")
# exit(1)


tensor_list = torch.load(pt_path)  # list[tensor]
print(len(tensor_list))
print(type(tensor_list))
print(tensor_list[0])
print(type(tensor_list[0]))

for idx, tensor in enumerate(tensor_list):
#    print(tensor)
#    exit(1)
    npy_path = os.path.join(save_dir, f'{idx}.npy')
    np.save(npy_path, tensor.numpy())

print(f"Saved {len(tensor_list)} .npy files to {save_dir}")

