# This is an example to show how to save the config file in another name.
# Though you can define the name of the saved config file, 
# it is recommended to use the same name in oneproject because it's easy for `cfgmanager.find` to find.
# Run this example by `python 2_save_config_another_name.py`

import hydra
from omegaconf import DictConfig

from cfgmanager import save

@hydra.main(version_base=None, config_path="conf/", config_name="config2")
def main_func(cfg: DictConfig) -> None:

    # 1. generate a path to save config and other data
    save_path = "./data/002/"

    # 2. save config
    #    Then you can find the saved config in save_path
    save(cfg, save_path, cfg_save_filename="parameters.yaml")

    # 3. save other data
    #    ...

    # 4. run this example again, you will get a new saved config file and an old config.
    #    Don't have to worry about losing the old config file.

if __name__ == "__main__": 
    main_func()