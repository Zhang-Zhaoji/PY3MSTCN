# import something...
import json
import datetime
from typing import Any, Dict, List, Optional, Union

def load_cfg(cfg_file: str) -> Dict[str, Any]:
    with open(cfg_file, "r") as f:
        cfg = json.load(f)
    return cfg

class Config(object):
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.cfg = load_cfg(cfg_file)
        self.model_name = self.cfg["model_name"] if "model_name" in self.cfg else f"default-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.save_model = self.cfg["save_model"] if "save_model" in self.cfg else True
        self.max_save_model = self.cfg["max_save_model"] if "max_save_model" in self.cfg else 3
        self.val_interval = self.cfg["val_interval"] if "val_interval" in self.cfg else 1
        self.epoch = self.cfg["epoch"] if "epoch" in self.cfg else 100
        self.lr = self.cfg["lr"] if "lr" in self.cfg else 0.001
        self.lr_decay = self.cfg["lr_decay"] if "lr_decay" in self.cfg else 0.95
        self.lr_decay_step = self.cfg["lr_decay_step"] if "lr_decay_step" in self.cfg else 10
        self.batch_size = self.cfg["batch_size"] if "batch_size" in self.cfg else 32
        self.num_workers = self.cfg["num_workers"] if "num_workers" in self.cfg else 4
        self.weight_decay = self.cfg["weight_decay"] if "weight_decay" in self.cfg else 0.0001
        self.optimizer = self.cfg["optimizer"] if "optimizer" in self.cfg else "Adam"
        self.scheduler = self.cfg["scheduler"] if "scheduler" in self.cfg else "StepLR"
        self.result_path = self.cfg["result_path"] if "result_path" in self.cfg else "./rsts"
        self.resume = self.cfg["resume"] if "resume" in self.cfg else ""
        self.num_stages = self.cfg["num_stages"] if "num_stages" in self.cfg else 3
        self.num_layers = self.cfg["num_layers"] if "num_layers" in self.cfg else 3
        self.num_f_maps = self.cfg["num_f_maps"] if "num_f_maps" in self.cfg else 64
        self.dim = self.cfg["dim"] if "dim" in self.cfg else 2048
        self.num_classes = self.cfg["num_classes"] if "num_classes" in self.cfg else 10
    def __getitem__(self, key):
        return self.cfg[key]
    
    def __setitem__(self, key, value):
        self.cfg[key] = value

    def __contains__(self, key):
        return key in self.cfg

def main():
    test_config_path = 'MyMSTCN/cfgs/default.json'
    cfg_json = json.load(open(test_config_path))
    print(cfg_json)
    cfg = Config(test_config_path)
    print(cfg)
    from model import build_from_cfg
    model = build_from_cfg(cfg)
    params = model.parameters()
    for param in params:
        print(type(param), param.size())


if __name__ == "__main__":
    main()
