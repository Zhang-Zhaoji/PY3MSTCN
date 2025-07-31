import torch
import torch.nn as nn  
import datetime
import torch.optim as optim
import tqdm
import os
import pathlib
from logger import Logger 
from config import Config
from model import build_from_cfg, MSTCN_criterion
from utils import AverageMeter
from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self,   model_cfg:Config,
                         train_loader:torch.utils.data.DataLoader, 
                         val_loader:torch.utils.data.DataLoader,
                         criterion:nn.Module = MSTCN_criterion, 
                         device:torch.device = None, 
                         logfile_dest:str=None,
                         model_dest:str=None,
                         wandb_project:str=None,
                         wandb_entity:str=None,
                         resume_model_path:str=None,
                         resume_optimizer_path:str=None
                         ):
        self.model_cfg = model_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.save_model = self.model_cfg.save_model
        self.max_save_model = self.model_cfg.max_save_model
        self.val_interval = self.model_cfg.val_interval
        self.result_path = self.model_cfg.result_path

        self.model_dest = model_dest if model_dest else self.result_path
        self.model_name = self.model_cfg.model_name
        self.model_path = os.path.join(self.model_dest, self.model_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        self.logger = Logger(self.model_cfg.cfg, logfile_dest= logfile_dest, wandb_project= wandb_project, wandb_entity= wandb_entity) 
        self.logger.info("Training model with config: ")
        self.logger.info(self.model_cfg.cfg)
        self.logger.info("Using device: ")
        self.logger.info(self.device)
        self.logger.info("Loading model")

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")

        
        try:
            self.model = build_from_cfg(self.model_cfg)
            self.model.to(self.device)
        except Exception as e:
            self.logger.error("Error loading model: ")
            self.logger.error(e)
            quit(f"Error loading model with cfg:{self.model_cfg.cfg}")
        self.logger.info("loading model successfully")

        self.optimizer = None
        if self.model_cfg.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.model_cfg.lr, weight_decay=self.model_cfg.weight_decay)
        else:
            raise ValueError(f"optimizer {self.model_cfg.optimizer} not supported now")
        self.logger.info(f"building optimizer with optimizer:{self.model_cfg.optimizer}")

        self.scheduler = None
        if self.model_cfg.scheduler == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.model_cfg.lr_decay_step, gamma=self.model_cfg.lr_decay)
        else:
            raise ValueError(f"scheduler {self.model_cfg.scheduler} not supported now")
        self.logger.info(f"building scheduler with scheduler:{self.model_cfg.scheduler}")
        if resume_model_path or resume_optimizer_path:
            if resume_model_path and resume_optimizer_path:
                self.resume()
                self.logger.info("Resume success.")
            else:
                raise ValueError(f"resume_model_path: {resume_model_path} or resume_optimizer_path:{resume_model_path} not provided")
    def resume(self):
        self.logger.info("Resuming model")
        assert self.resume_model_path and self.resume_optimizer_path
        self.logger.info(f"resume model from {self.resume_model_path}")
        self.model.load_state_dict(torch.load(self.resume_model_path))
        self.logger.info(f"loading optimizer from {self.resume_optimizer_path}")
        self.optimizer.load_state_dict(torch.load(self.resume_optimizer_path))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.model_cfg.lr_decay_step, gamma=self.model_cfg.lr_decay)

    def train(self, epoch:int, log_interval:int=10):
        self.model.train()
        self.logger.info(f"Training epoch {epoch}")
        
        train_metrics = AverageMeter("Average Training Metric")
        correct = 0
        total = 0
        for batch_idx, (data, target, mask) in tqdm.tqdm(enumerate(self.train_loader),total = len(self.train_loader)):
            data, target, mask = data.to(self.device), target.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data, mask)
            loss = self.criterion(output, target)
            train_metrics.update(loss.item(), data.size(0))
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(output[-1].data, 1)
            correct += ((predicted == target).float()*mask[:, 0, :].squeeze(1)).sum().item()
            total += torch.sum(mask[:, 0, :]).item()
            if batch_idx % log_interval == log_interval - 1:
                self.logger.info(f"Train Epoch: {epoch}, {str(train_metrics)}, Acc: {correct/total:.4f}, [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\t")
        self.logger.info(f"Train Epoch: {epoch}, {str(train_metrics)}, Acc: {correct/total:.4f}]\t")
        self.logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]}")
        self.scheduler.step()
        if self.save_model:
            
            if pathlib.exists(os.path.join(self.result_path, f'model_epoch_{epoch - self.max_save_model}.pth')):
                os.remove(os.path.join(self.result_path, f'model_epoch_{epoch - self.max_save_model}.pth'))   
            if pathlib.exists(os.path.join(self.result_path, f"optimizer_epoch_{epoch- self.max_save_model}.pth")):
                os.remove(os.path.join(self.result_path, f"optimizer_epoch_{epoch- self.max_save_model}.pth"))         
              
            torch.save(self.model.state_dict(), os.path.join(self.result_path, f"model_epoch_{epoch}.pth"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.result_path, f"optimizer_epoch_{epoch}.pth"))

    def validate(self, epoch:int, metric_function:callable, log_interval:int=20):
        self.model.eval()
        self.logger.info(f"Epoch: {epoch}")
        self.logger.info("Validation begins: ")
        self.logger.info(f"Validation set size: {len(self.val_loader.dataset)}")
        self.logger.info(f"Validation batch size: {self.val_loader.batch_size}")
        self.logger.info(f"Validation steps: {len(self.val_loader)}")
        val_metric = AverageMeter("Average Validation Metric")
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target, mask) in tqdm.tqdm(enumerate(self.val_loader),total=len(self.val_loader)):
                data, target, mask = data.to(self.device), target.to(self.device), mask.to(self.device)
                output = self.model(data, mask)
                metric = metric_function(output, target, mask)
                _, predicted = torch.max(output[-1].data, 1)
                correct += ((predicted == target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
                val_metric.update(metric.item(), data.size(0))
                if batch_idx % log_interval == log_interval-1:
                    self.logger.info(f"Validation step: {batch_idx} / {len(self.val_loader)}, {str(val_metric)}, Acc: {correct/total:.4f} ")
            self.logger.info(f"Validate Epoch: {epoch}, {str(val_metric)}, Acc: {correct/total:.4f}]\t")
            self.logger.info("Validation ends: ")


    def test(self, test_dataloader:torch.utils.data.DataLoader, metric_function:callable):
        self.model.eval()
        self.logger.info("Test begins: ")
        self.logger.info(f"Test set size: {len(test_dataloader.dataset)}")
        self.logger.info(f"Test batch size: {test_dataloader.batch_size}")
        self.logger.info(f"Test steps: {len(test_dataloader)}")
        test_metric = AverageMeter("Average Test Metric")
        with torch.no_grad():
            for batch_idx, (data, target, mask) in tqdm.tqdm(enumerate(test_dataloader),total = len(test_dataloader)):
                data, target, mask = data.to(self.device), target.to(self.device), mask.to(self.device)
                output = self.model(data, mask)
                metric = metric_function(output, target, mask)
                test_metric.update(metric.item(), data.size(0))
                if batch_idx % 10 == 0:
                    self.logger.info(f"Test step: {batch_idx} / {len(test_dataloader)}, " + str(test_metric))
        self.logger.info("Test ends: ")
        self.logger.info(f"Test metric: {test_metric}")
        return test_metric

    def predict(self, vid_list_file):
        self.model.eval()
        with torch.no_grad():
            self.logger.info("Prediction starts: ")
            file = open(vid_list_file, "r")
            list_of_videos = [line.strip() for line in file.readlines()]
            file.close()
            self.logger.info(f"Total number of videos: {len(list_of_videos)}")
            self.logger.info(f'{list_of_videos}...')
            # for vid in list_of_videos:
            #     features = np.load(features_path + vid.split('.')[0] + '.npy')
            #     features = features[:, ::sample_rate]
            #     input_x = torch.tensor(features, dtype=torch.float)
            #     input_x.unsqueeze_(0)
            #     input_x = input_x.to(device)
            #     predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
            #     _, predicted = torch.max(predictions[-1].data, 1)
            #     predicted = predicted.squeeze()
            #     recognition = []
            #     for i in range(len(predicted)):
            #         recognition.append(actions_dict[predicted[i].item()])
            #     self.logger.info(f"Video: {vid}, Recognition: {recognition}")
            # self.logger.info("Prediction ends: ")

    def train_loop(self, epochs:int = None, train_log_interval:int=50, val_log_interval:int=50):
        """
        main training loop
        """
        if epochs is None:
            epochs = self.epochs
        self.logger.info("Starting training loop")
        for idx in range(epochs):
            self.logger.info(f"Epoch: {idx}/{epochs}")
            self.logger.info("Training starts: ")
            self.train(idx, train_log_interval)
            if idx % self.val_interval == self.val_interval - 1:
                self.logger.info("Validation starts: ")
                self.validate(idx, val_log_interval)
                self.logger.info("Validation ends: ")
            self.logger.info("Training ends: ")




        