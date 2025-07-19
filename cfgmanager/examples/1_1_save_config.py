# This is a minimal example to show how to save the config file.
# This example is based on hydra, you have to know how to use hydra before reading this example.
# If you want to use more features of `cfgmanager.save`, please refer to the docstring.
# Run this example by `python 1_save_config.py`

import hydra
from omegaconf import DictConfig

from cfgmanager import save

@hydra.main(version_base=None, config_path="conf/", config_name="config1")
def main_func(cfg: DictConfig) -> None:

    # 1. generate a path to save config and other data
    save_path = "./data/001/"

    # 2. save config
    #    Then you can find the saved config in save_path
    save(cfg, save_path)

    # 3. save other data
    #    ...

    # 4. run this example again, you will get a new saved config file and an old config.
    #    Don't have to worry about losing the old config file.

if __name__ == "__main__": 
    main_func()