import wandb
import datetime
import json

class Logger:
    def __init__(self, cfg, logfile_dest=None, wandb_project=None, wandb_entity=None) -> None:
        self.logfile_dest = logfile_dest if logfile_dest else f"log{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
        self.wandb_project = wandb_project if wandb_project else None
        self.wandb_entity = wandb_entity if wandb_entity else None
        self.wandb = None
        if wandb_project is not None and wandb_entity is not None:
            self.wandb = wandb.init(project=self.wandb_project, entity=self.wandb_entity)
            self.wandb.config.update(cfg)
        self.logfile = open(self.logfile_dest, "a")
        # 在__init__方法中改进配置写入
        self.logfile.write(json.dumps(cfg, indent=2) + "\n")

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {msg}"
        try:
            self.logfile.write(log_entry + "\n")
            self.logfile.flush()  # 确保立即写入
        except Exception as e:
            print(f"Failed to write to log file: {e}")
        
        if self.wandb:
            try:
                # 如果msg是字典，使用Wandb的log方法正确记录
                if isinstance(msg, dict):
                    self.wandb.log(msg)
                else:
                    # 否则记录为文本
                    self.wandb.log({"message": msg})
            except Exception as e:
                print(f"Failed to log to Wandb: {e}")

    def print(self, msg):
        print(msg)
        self.log(msg)

    def __del__(self):
        try:
            if hasattr(self, 'logfile') and self.logfile:
                self.logfile.close()
        except Exception as e:
            print(f"Error closing log file: {e}")
        
        try:
            if hasattr(self, 'wandb') and self.wandb:
                self.wandb.finish()
        except Exception as e:
            print(f"Error finishing Wandb: {e}")


    def info(self, msg):
        self.print(f"[INFO] {msg}")
    
    def warning(self, msg):
        self.print(f"[WARNING] {msg}")
        
    def error(self, msg):
        self.print(f"[ERROR] {msg}")