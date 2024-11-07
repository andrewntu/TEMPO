#/bin/python

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from torch.utils.data import Subset
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.TEMPO import TEMPO
# from models.T5 import T54TS
from models.ETSformer import ETSformer
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf


config = {
    "description": "TEMPO",
    "model_id": "ETTh2_TEMPO_multi-test",
    "checkpoints": "./checkpoints",
    "task_name": "long_term_forecast",
    "prompt": 1,
    "num_nodes": 1,
    "seq_len": 336,
    "pred_len": 96,
    "label_len": 96,
    "decay_fac": 0.5,
    "learning_rate": 0.001,
    "batch_size": 256,
    "num_workers": 0,
    "train_epochs": 10,
    "lradj": "type3",
    "patience": 5,
    "gpt_layers": 6,
    "is_gpt": 1,
    "e_layers": 3,
    "d_model": 768,
    "n_heads": 4,
    "d_ff": 768,
    "dropout": 0.3,
    "enc_in": 7,
    "c_out": 1,
    "patch_size": 16,
    "kernel_size": 25,
    "loss_func": "mse",
    "pretrain": 1,
    "freeze": 1,
    "model": "TEMPO",
    "stride": 8,
    "max_len": -1,
    "hid_dim": 16,
    "tmax": 20,
    "itr": 3,
    "cos": 1,
    "equal": 1,
    "pool": False,
    "no_stl_loss": False,
    "stl_weight": 0.001,
    "config_path": "./configs/custom_dataset.yml",
    "datasets": "ETTm1,ETTh1,ETTm2,electricity,traffic,weather",
    "target_data": "Custom",
    "use_token": 0,
    "electri_multiplier": 1,
    "traffic_multiplier": 1,
    "embed": "timeF",
    "percent": 100,
    "model_id": "./checkpoints/TEMPO-80M_v1"
}
SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}

cfg = OmegaConf.create(config)
# Save the configuration to a YAML file
with open("./configs/etth2_config.yml", "w") as f:
    OmegaConf.save(cfg, f)

def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config
    
config = get_init_config(cfg.config_path)


setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(cfg.model_id, 336, cfg.label_len, cfg.pred_len,
                                                                    cfg.d_model, cfg.n_heads, cfg.e_layers, cfg.gpt_layers,
                                                                    cfg.d_ff, cfg.embed, 0)
path = os.path.join(cfg.checkpoints, setting)

if not os.path.exists(path):
    os.makedirs(path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

# print(config,cfg.target_data)

cfg.data = config['datasets'][cfg.target_data].data
cfg.root_path = config['datasets'][cfg.target_data].root_path
cfg.data_path = config['datasets'][cfg.target_data].data_path
cfg.data_name = config['datasets'][cfg.target_data].data_name
cfg.features = config['datasets'][cfg.target_data].features
cfg.freq = config['datasets'][cfg.target_data].freq
cfg.target = config['datasets'][cfg.target_data].target
cfg.embed = config['datasets'][cfg.target_data].embed
cfg.percent = config['datasets'][cfg.target_data].percent
cfg.lradj = config['datasets'][cfg.target_data].lradj
if cfg.freq == 0:
    cfg.freq = 'h'
test_data, test_loader = data_provider(cfg, 'pred')


# print(config['datasets'],cfg.target_data)