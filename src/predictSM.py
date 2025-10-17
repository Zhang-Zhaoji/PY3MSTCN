import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from config import Config
from trainer_Sal import Trainer
from torch.utils.data import Dataset, DataLoader
from datasetSM import CausalityInTrafficAccident, collate_fn
import argparse
import json
import torch
torch.manual_seed(7802)
torch.cuda.manual_seed(7802)
torch.cuda.manual_seed_all(7802)


def main(args:argparse.ArgumentParser):
    p = vars(args)
    p['len_sequence'] = 416 #208
    p['fps'] = 25
    p['vid_length'] = p['len_sequence'] * 8 / p['fps']
    print(p)
    defined_configs = json.load(open(p['cfg_path'], 'r'))
    for k, v in defined_configs.items():
        p[k] = v

    if args.mode == 'train':
        dataset_train = CausalityInTrafficAccident(p, split='train')
        dataset_val   = CausalityInTrafficAccident(p, split='val', test_mode=True)
        dataset_test  = []
        dataloader_train = DataLoader(dataset_train, batch_size=p['batch_size'], shuffle=True, num_workers=p['num_workers'],collate_fn=collate_fn,pin_memory=True)
        dataloader_val = DataLoader(dataset_val, batch_size=p['batch_size'], num_workers=p['num_workers'],collate_fn=collate_fn,pin_memory=True)
        dataloader_test = None
    elif args.mode == 'test':
        dataset_train = []
        dataset_val   = []
        dataset_test  = CausalityInTrafficAccident(p, split='test', test_mode=True)
        dataloader_train, dataloader_val = None, None
        dataloader_test = DataLoader(dataset_test, batch_size=p['batch_size'], num_workers=p['num_workers'],collate_fn=collate_fn,pin_memory=True)
    elif args.mode == 'predict':
        dataset_train = []
        dataset_val   = []
        dataset_test  = CausalityInTrafficAccident(p, split='test', test_mode=True)
        dataloader_train, dataloader_val = None, None
        dataloader_predict = DataLoader(dataset_test, batch_size=p['batch_size'], num_workers=p['num_workers'],collate_fn=collate_fn,pin_memory=True)
    else:
        raise ValueError('mode must be train or test or predict')
    
    
    print(f"train/validation/test dataset size{len(dataset_train), len(dataset_val), len(dataset_test)}")
    # print(dataloader_test.dataset[0])
    # exit(1)

    main_Trainer = Trainer(model_cfg=Config(p['cfg_path']),
                           train_loader=dataloader_train,
                           val_loader=dataloader_val,
                           logfile_dest=p['logfile_dest'],
                           model_dest=p['model_dest'],
                           wandb_project=p['wandb_project'],
                           wandb_entity=p['wandb_entity'],
                           resume_model_path=p['resume_model_path'],
                           resume_optimizer_path=p['resume_optimizer_path']
                           )
    # test_result_dict = main_Trainer.test(test_dataloader=dataloader_test, metric_function=main_Trainer.criterion)
    rst = main_Trainer.predict(test_dataloader=dataloader_predict, save_path=p['prediction_save_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='cfgs/v2-2-128.json')
    parser.add_argument('--logfile_dest', type=str, default='./logs/test_log')
    parser.add_argument('--model_dest', type=str, default='./model')
    parser.add_argument('--wandb_project', type=str, default='MyMSTCN')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--train_loader', type=str, default='./train_loader.pkl')
    parser.add_argument('--num_workers', type=int, default=1) # 4
    parser.add_argument('--batch_size', type=int, default=1) # 16
    parser.add_argument('--mode',type=str, default='predict', choices=['train', 'test', 'predict'])
    parser.add_argument('--resume_model_path', type=str, default=r'model/bestSal2/2025-08-18-21-26-32/best_model.pth')
    parser.add_argument('--resume_optimizer_path', type=str, default='')
    parser.add_argument('--feature', type=str, default="SMfeature")
    parser.add_argument('--feature_folder', type=str, default="Saldataset")
    parser.add_argument('--input_size', type=int, default=25088)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_segments', type=int, default=4)
    parser.add_argument('--new_length', type=int, default=1)      
    parser.add_argument('--feed_type', type=str, default="multi-label",choices=["multi-label", "detection", "classification"])
    parser.add_argument('--dataset_ver', type=str, default='Mar9th')
    parser.add_argument('--prediction_save_path', type=str, default='test_pred/')
    args = parser.parse_args()
    main(args)
