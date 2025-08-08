import torch
import torch.nn as nn  
import datetime
import torch.optim as optim
import tqdm
import os
import pathlib
from logger import Logger 
from config import Config
from model import build_from_cfg, ActionSegmentationLoss
# 在 trainer.py 文件开头添加导入
from utils import AverageMeter, compute_exact_iou, compute_temporalIoU
from torch.utils.tensorboard import SummaryWriter
import time
# from timm.utils.model_ema import ModelEmaV2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



class Trainer(object):
    def __init__(self,   model_cfg:Config,
                         train_loader:torch.utils.data.DataLoader, 
                         val_loader:torch.utils.data.DataLoader,
                         criterion:nn.Module = ActionSegmentationLoss(), 
                         device:torch.device = None, 
                         logfile_dest:str=None,
                         model_dest:str=None,
                         wandb_project:str=None,
                         wandb_entity:str=None,
                         resume_model_path:str=None,
                         resume_optimizer_path:str=None,
                         ):
        self.model_cfg = model_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.save_model = self.model_cfg.save_model
        self.max_save_model = self.model_cfg.max_save_model
        self.val_interval = self.model_cfg.val_interval
        self.result_path = self.model_cfg.result_path
        self.epochs  = self.model_cfg.epoch
        self.num_stages = self.model_cfg.num_stages
        self.num_layers = self.model_cfg.num_layers
        


        self.model_dest = model_dest if model_dest else self.result_path
        pathlib.Path(self.model_dest).mkdir(parents=True, exist_ok=True)
        self.model_name = self.model_cfg.model_name
        self.model_path = os.path.join(self.model_dest, self.model_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.model_path, "tb_logs" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        pathlib.Path(self.model_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(model_cfg, logfile_dest= logfile_dest, wandb_project= wandb_project, wandb_entity= wandb_entity) 
        self.logger.info("Training model with config: ")
        self.logger.info(self.model_cfg.cfg)
        self.logger.info("Using device: ")
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
        self.logger.info("Loading model")
        
        try:
            self.model = build_from_cfg(self.model_cfg,False)
            #self.ema = ModelEmaV2(self.model, decay=0.9999)
            self.model.to(self.device)
            #self.ema.module.to(self.device)
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
        dummy = torch.randn(1, *self.train_loader.dataset[0]['feature'].shape).to(self.device)
        self.tb_writer.add_graph(self.model, dummy)
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
        cause_iou_metrics = AverageMeter("Average Cause IoU")
        effect_iou_metrics = AverageMeter("Average Effect IoU")
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx, (data, target, mask) in enumerate(tqdm.tqdm(self.train_loader,total = len(self.train_loader))):
            data, target, mask = data.to(self.device), target.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target, mask)
            train_metrics.update(loss.item(), data.size(0))
            # with torch.autograd.detect_anomaly():
            loss.backward()
            self.optimizer.step()
            # self.ema.update(self.model)
            _, predicted = torch.max(output[-1].data, 1)
            correct += ((predicted == target)*mask).float().sum().item()
            total += torch.sum(mask).item()

            # 计算cause和effect的IoU
            cause_iou, effect_iou = compute_exact_iou(output[-1], target, mask, predtype='both')
            for i in range(len(cause_iou)):
                cause_iou_metrics.update(cause_iou[i].item(), 1)
                effect_iou_metrics.update(effect_iou[i].item(), 1)

            if batch_idx % log_interval == log_interval - 1:
                self.logger.info(f"Train Epoch: {epoch}, {str(train_metrics)}, Acc: {correct/total:.4f}, Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f}, [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\t")

        self.logger.info(f"Train Epoch: {epoch}, {str(train_metrics)}, Acc: {correct/total:.4f}, Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f}]\t")
        self.logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]}")
        self.scheduler.step()
        epoch_time = time.time() - epoch_start 

        # 计算IoU阈值统计
        cause_0129_iou = compute_temporalIoU(cause_iou_metrics.val)  # 0129 means 0.1 to 0.9
        effect_0129_iou = compute_temporalIoU(effect_iou_metrics.val)

        self.logger.info(f"Cause  0.1 ~ 0.9 iou: {cause_0129_iou}")
        self.logger.info(f"Effect 0.1 ~ 0.9 iou: {effect_0129_iou}")
        self.logger.info(f"Both   0.1 ~ 0.9 iou: {(cause_0129_iou+effect_0129_iou)/2}")

        # 特别关注的阈值统计
        thresholds = [0.1, 0.3, 0.5, 0.9]
        threshold_indices = [0, 2, 4, 8]  # 对应于compute_temporalIoU返回的索引

        self.logger.info("Key IoU Thresholds:")
        self.logger.info("Cause IoU:")
        for i, idx in enumerate(threshold_indices):
            self.logger.info(f"  IoU > {thresholds[i]:.1f}: {cause_0129_iou[idx]:.4f}")

        self.logger.info("Effect IoU:")
        for i, idx in enumerate(threshold_indices):
            self.logger.info(f"  IoU > {thresholds[i]:.1f}: {effect_0129_iou[idx]:.4f}")

        self.logger.info(f"Epoch time: {epoch_time:.2f}s")

        # 写入TensorBoard
        self.tb_writer.add_scalar("Train/Loss_epoch", train_metrics.avg, epoch)
        self.tb_writer.add_scalar("Train/Accuracy_epoch", correct / total, epoch)
        self.tb_writer.add_scalar("Train/Cause_IoU_epoch", cause_iou_metrics.avg, epoch)
        self.tb_writer.add_scalar("Train/Effect_IoU_epoch", effect_iou_metrics.avg, epoch)
        self.tb_writer.add_scalar("Train/LR", self.scheduler.get_last_lr()[0], epoch)
        self.tb_writer.add_scalar("Train/Time", epoch_time, epoch)
        self.tb_writer.add_scalar("Train/SamplesPerSec", total / epoch_time, epoch)

        # 写入关键阈值统计到TensorBoard
        for i, idx in enumerate(threshold_indices):
            self.tb_writer.add_scalar(f"Train/Cause_IoU_over_{int(thresholds[i]*10)}", 
                                    cause_0129_iou[idx], epoch)
            self.tb_writer.add_scalar(f"Train/Effect_IoU_over_{int(thresholds[i]*10)}", 
                                    effect_0129_iou[idx], epoch)
            self.tb_writer.add_scalar(f"Train/Combined_IoU_over_{int(thresholds[i]*10)}", 
                                    (cause_0129_iou[idx] + effect_0129_iou[idx]) / 2, epoch)

        if self.save_model: 
            if os.path.exists(os.path.join(self.result_path, f"checkpoint{epoch-self.max_save_model}.pth")):
                os.remove(os.path.join(self.result_path, f"checkpoint{epoch-self.max_save_model}.pth"))
            torch.save({'model_state_dict':self.model.state_dict(),
                        # 'ema_state_dict':self.ema.module.state_dict(),
                        'optimizer':self.optimizer.state_dict()}, os.path.join(self.result_path, f"checkpoint{epoch}.pth"))

    def validate(self, epoch:int, metric_function:callable, log_interval:int=20):
        self.model.eval()
        # self.ema.module.eval()
        self.logger.info(f"Epoch: {epoch}")
        self.logger.info("Validation begins: ")
        self.logger.info(f"Validation set size: {len(self.val_loader.dataset)}")
        self.logger.info(f"Validation batch size: {self.val_loader.batch_size}")
        self.logger.info(f"Validation steps: {len(self.val_loader)}")

        val_metric = AverageMeter("Average Validation Metric")
        cause_iou_metrics = AverageMeter("Average Validation Cause IoU")
        effect_iou_metrics = AverageMeter("Average Validation Effect IoU")
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(tqdm.tqdm(self.val_loader,total=len(self.val_loader))):
                data, target, mask = data.to(self.device), target.to(self.device), mask.to(self.device)
                # output = self.ema.module(data)
                output = self.model(data)
                metric = metric_function(output, target, mask)
                _, predicted = torch.max(output[-1].data, 1)  # 使用最后一个stage的输出
                correct += ((predicted == target)*mask).float().sum().item()
                total += torch.sum(mask).item()
                val_metric.update(metric.item(), data.size(0))

                # 计算IoU
                cause_iou, effect_iou = compute_exact_iou(output[-1], target, mask, predtype='both')
                for i in range(len(cause_iou)):
                    cause_iou_metrics.update(cause_iou[i].item(), 1)
                    effect_iou_metrics.update(effect_iou[i].item(), 1)

                if batch_idx % log_interval == log_interval-1:
                    self.logger.info(f"Validation step: {batch_idx} / {len(self.val_loader)}, {str(val_metric)}, Acc: {correct/total:.4f}, Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f} ")

            # 计算IoU阈值统计
            cause_0129_iou = compute_temporalIoU(cause_iou_metrics.val)
            effect_0129_iou = compute_temporalIoU(effect_iou_metrics.val)

            self.logger.info(f"Validate Epoch: {epoch}, {str(val_metric)}, Acc: {correct/total:.4f}, Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f}]\t")

            # 关键阈值统计
            thresholds = [0.1, 0.3, 0.5, 0.9]
            threshold_indices = [0, 2, 4, 8]

            self.logger.info("Validation Key IoU Thresholds:")
            self.logger.info("Cause IoU:")
            for i, idx in enumerate(threshold_indices):
                self.logger.info(f"  IoU > {thresholds[i]:.1f}: {cause_0129_iou[idx]:.4f}")

            self.logger.info("Effect IoU:")
            for i, idx in enumerate(threshold_indices):
                self.logger.info(f"  IoU > {thresholds[i]:.1f}: {effect_0129_iou[idx]:.4f}")

            self.logger.info("Combined IoU:")
            for i, idx in enumerate(threshold_indices):
                combined = (cause_0129_iou[idx] + effect_0129_iou[idx]) / 2
                self.logger.info(f"  IoU > {thresholds[i]:.1f}: {combined:.4f}")

        # 写入TensorBoard
        self.tb_writer.add_scalar("Val/Loss_epoch", val_metric.avg, epoch)
        self.tb_writer.add_scalar("Val/Accuracy_epoch", correct / total, epoch)
        self.tb_writer.add_scalar("Val/Cause_IoU_epoch", cause_iou_metrics.avg, epoch)
        self.tb_writer.add_scalar("Val/Effect_IoU_epoch", effect_iou_metrics.avg, epoch)

        # 写入关键阈值统计到TensorBoard
        for i, idx in enumerate(threshold_indices):
            self.tb_writer.add_scalar(f"Val/Cause_IoU_over_{int(thresholds[i]*10)}", 
                                    cause_0129_iou[idx], epoch)
            self.tb_writer.add_scalar(f"Val/Effect_IoU_over_{int(thresholds[i]*10)}", 
                                    effect_0129_iou[idx], epoch)
            self.tb_writer.add_scalar(f"Val/Combined_IoU_over_{int(thresholds[i]*10)}", 
                                    (cause_0129_iou[idx] + effect_0129_iou[idx]) / 2, epoch)

        self.logger.info("Validation ends: ")

    def test(self, test_dataloader:torch.utils.data.DataLoader, metric_function:callable):
        self.model.eval()
        self.logger.info("Test begins: ")
        self.logger.info(f"Test set size: {len(test_dataloader.dataset)}")
        self.logger.info(f"Test batch size: {test_dataloader.batch_size}")
        self.logger.info(f"Test steps: {len(test_dataloader)}")

        test_metric = AverageMeter("Average Test Metric")
        cause_iou_metrics = AverageMeter("Average Test Cause IoU")
        effect_iou_metrics = AverageMeter("Average Test Effect IoU")
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(tqdm.tqdm(test_dataloader, total=len(test_dataloader))):
                data, target, mask = data.to(self.device), target.to(self.device), mask.to(self.device)
                output = self.model(data)
                metric = metric_function(output, target, mask)
                test_metric.update(metric.item(), data.size(0))

                _, predicted = torch.max(output[-1].data, 1)
                correct += ((predicted == target)*mask).float().sum().item()
                total += torch.sum(mask).item()

                # 计算IoU
                cause_iou, effect_iou = compute_exact_iou(output[-1], target, mask, predtype='both')
                for i in range(len(cause_iou)):
                    cause_iou_metrics.update(cause_iou[i].item(), 1)
                    effect_iou_metrics.update(effect_iou[i].item(), 1)

                if batch_idx % 10 == 0:
                    self.logger.info(f"Test step: {batch_idx} / {len(test_dataloader)}, {str(test_metric)}, Acc: {correct/total:.4f}, Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f}")

            # 计算IoU阈值统计
            cause_0129_iou = compute_temporalIoU(cause_iou_metrics.val)
            effect_0129_iou = compute_temporalIoU(effect_iou_metrics.val)

            self.logger.info(f"Test Results: {str(test_metric)}, Acc: {correct/total:.4f}, Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f}")

            # 关键阈值统计
            thresholds = [0.1, 0.3, 0.5, 0.9]
            threshold_indices = [0, 2, 4, 8]

            self.logger.info("Test Key IoU Thresholds:")
            self.logger.info("Cause IoU:")
            for i, idx in enumerate(threshold_indices):
                self.logger.info(f"  IoU > {thresholds[i]:.1f}: {cause_0129_iou[idx]:.4f}")

            self.logger.info("Effect IoU:")
            for i, idx in enumerate(threshold_indices):
                self.logger.info(f"  IoU > {thresholds[i]:.1f}: {effect_0129_iou[idx]:.4f}")

            self.logger.info("Combined IoU:")
            for i, idx in enumerate(threshold_indices):
                combined = (cause_0129_iou[idx] + effect_0129_iou[idx]) / 2
                self.logger.info(f"  IoU > {thresholds[i]:.1f}: {combined:.4f}")

        self.logger.info("Test ends: ")
        return {
            'loss': test_metric.avg,
            'accuracy': correct / total,
            'cause_iou': cause_iou_metrics.avg,
            'effect_iou': effect_iou_metrics.avg,
            'cause_iou_thresholds': cause_0129_iou,
            'effect_iou_thresholds': effect_0129_iou
        }

    def predict(self, test_dataloader: torch.utils.data.DataLoader, save_path: str = None):
        """
        对测试数据进行预测并可选择保存结果

        params:
        test_dataloader: 测试数据加载器
        save_path: 可选，保存预测结果的路径
        """
        self.model.eval()
        self.logger.info("Prediction begins: ")
        self.logger.info(f"Prediction set size: {len(test_dataloader.dataset)}")
        self.logger.info(f"Prediction batch size: {test_dataloader.batch_size}")
        self.logger.info(f"Prediction steps: {len(test_dataloader)}")

        predictions = []
        cause_iou_metrics = AverageMeter("Average Prediction Cause IoU")
        effect_iou_metrics = AverageMeter("Average Prediction Effect IoU")

        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(tqdm.tqdm(test_dataloader, total=len(test_dataloader))):
                data, target, mask = data.to(self.device), target.to(self.device), mask.to(self.device)
                output = self.model(data)

                # 获取预测结果
                _, predicted = torch.max(output[-1].data, 1)

                # 保存预测结果
                batch_predictions = {
                    'predicted': predicted.cpu(),
                    'target': target.cpu(),
                    'mask': mask.cpu(),
                    'output': output[-1].cpu()
                }
                predictions.append(batch_predictions)

                # 计算IoU（如果有真实标签）
                if target is not None and mask is not None:
                    cause_iou, effect_iou = compute_exact_iou(output[-1], target, mask, predtype='both')
                    for i in range(len(cause_iou)):
                        cause_iou_metrics.update(cause_iou[i].item(), 1)
                        effect_iou_metrics.update(effect_iou[i].item(), 1)

                if batch_idx % 10 == 0:
                    if target is not None and mask is not None:
                        self.logger.info(f"Prediction step: {batch_idx} / {len(test_dataloader)}, Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f}")
                    else:
                        self.logger.info(f"Prediction step: {batch_idx} / {len(test_dataloader)}")

            # 如果有真实标签，计算总体IoU统计
            if len(cause_iou_metrics.val) > 0 and len(effect_iou_metrics.val) > 0:
                cause_0129_iou = compute_temporalIoU(cause_iou_metrics.val)
                effect_0129_iou = compute_temporalIoU(effect_iou_metrics.val)

                self.logger.info(f"Prediction Results - Cause IoU: {cause_iou_metrics.avg:.4f}, Effect IoU: {effect_iou_metrics.avg:.4f}")

                # 关键阈值统计
                thresholds = [0.1, 0.3, 0.5, 0.9]
                threshold_indices = [0, 2, 4, 8]

                self.logger.info("Prediction Key IoU Thresholds:")
                self.logger.info("Cause IoU:")
                for i, idx in enumerate(threshold_indices):
                    self.logger.info(f"  IoU > {thresholds[i]:.1f}: {cause_0129_iou[idx]:.4f}")

                self.logger.info("Effect IoU:")
                for i, idx in enumerate(threshold_indices):
                    self.logger.info(f"  IoU > {thresholds[i]:.1f}: {effect_0129_iou[idx]:.4f}")

                self.logger.info("Combined IoU:")
                for i, idx in enumerate(threshold_indices):
                    combined = (cause_0129_iou[idx] + effect_0129_iou[idx]) / 2
                    self.logger.info(f"  IoU > {thresholds[i]:.1f}: {combined:.4f}")

        # 保存预测结果
        if save_path:
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(predictions, os.path.join(save_path, "predictions.pth"))
            self.logger.info(f"Predictions saved to {os.path.join(save_path, 'predictions.pth')}")

        self.logger.info("Prediction ends: ")
        return predictions

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
            self.logger.info("Training ends: ")
            if idx % self.val_interval == self.val_interval - 1:
                self.logger.info("Validation starts: ")
                self.validate(idx, self.criterion, val_log_interval)
                self.logger.info("Validation ends: ")
            if idx % 5 == 4:   # 每 5 个 epoch 写一次
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.tb_writer.add_histogram(f"Grad/{name}", param.grad, idx)
                    self.tb_writer.add_histogram(f"Weight/{name}", param, idx)




        