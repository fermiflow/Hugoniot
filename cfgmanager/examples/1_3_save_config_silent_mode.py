# This is an example to show how to save the config file in silent mode.
# Run this example by `python 1_save_config.py`

import hydra
from omegaconf import DictConfig

from cfgmanager import save

@hydra.main(version_base=None, config_path="conf/", config_name="config3")
def main_func(cfg: DictConfig) -> None:

    # 1. generate a path to save config and other data
    save_path = "./data/003/"

    # 2. save config in silent mode
    cfg_filename = save(cfg, save_path, silent_mode=True)
    print(f"saved config file to: {cfg_filename}")

if __name__ == "__main__": 
    main_func()