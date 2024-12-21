import hydra
import os 
from omegaconf import DictConfig
from importlib import import_module
from pathlib import Path
import logging
import torch
from utils.random_seed import setup_seed
import warnings
import torch.multiprocessing as mp 


warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

config_path = '/home/ljt21/UniADRS/configs/training&infer_img_folder.yaml'

@hydra.main(config_path)
def main(cfg: DictConfig) -> None:
    
    setup_seed(cfg.random_seed)

    working_dir = str(Path.cwd())
    logging.info(f"The current working directory is {working_dir}")
    
    runner_module_cfg = cfg.runner_module
    module_path, attr_name = runner_module_cfg.split(" - ")
    module = import_module(module_path)
    runner_module = getattr(module, attr_name)
    runner = runner_module(cfg, working_dir)

    runner.run()


if __name__ == "__main__":
    main()