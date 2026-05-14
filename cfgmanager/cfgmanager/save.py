import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf, open_dict

from cfgmanager.utils import *

def save(cfg: DictConfig,
         path: str,
         cfg_save_filename: str = "config.yaml",
         auto_save_path: bool = True,
         auto_save_time: bool = True,
         auto_save_time_format: str = "%Y-%m-%d_%H:%M:%S",
         silent_mode: bool = False,
    ) -> str:
    """
        Save config yaml file to path.
        INPUT:
            cfg: (omegaconf.DictConfig), config.
            path: (str), file path, save function will save config to this path.
            cfg_save_filename: (str), config file name. Default is "config.yaml".
            auto_save_path: (bool), if save path to config file. Default is True.
            auto_save_time: (bool), if save time to config. Default is True.
            auto_save_time_format: (str), auto save time format. Default is "%Y-%m-%d_%H:%M:%S".
            silent_mode: (bool), if run in silent mode. Default is False.
        OUTPUT:
            cfg_filename: (str), saved config file name.
    """

    # auto save keys
    auto_save_time_key = "auto_save_time"
    auto_save_path_key = "auto_save_path"
    auto_save_call_path_key = "call_path"
    auto_save_cfg_path_key = "cfg_path"

    # get current time
    now = datetime.now()
    current_time_str = now.strftime(auto_save_time_format)

    # create path
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        os.makedirs(path)

    # get cfg filename
    if cfg_save_filename.endswith(".yaml"):
        cfg_filename = os.path.join(path, cfg_save_filename)
    else:
        cfg_filename = os.path.join(path, cfg_save_filename + ".yaml")

    # rename old config file (if exists)
    if os.path.isfile(cfg_filename):
        cfg_old = OmegaConf.load(cfg_filename)
        if auto_save_time_key in cfg_old:
            cfg_old_filename = os.path.join(path, cfg_filename[:-5] + "_old_" + cfg_old[auto_save_time_key] + ".yaml")
        else:
            cfg_old_filename = os.path.join(path, cfg_filename[:-5] + "_old_" + current_time_str + ".yaml")
        os.rename(cfg_filename, cfg_old_filename)
        if not silent_mode:
            print(f"{YELLOW}rename old config to:{RESET}")
            print(f'File "{cfg_old_filename}"')

    # auto save time
    if auto_save_time:
        if auto_save_time_key in cfg:
            old_time_str = cfg[auto_save_time_key]
            if old_time_str != current_time_str:
                cfg[auto_save_time_key] = current_time_str
                if not silent_mode:
                    print(f"{YELLOW}change{RESET} {auto_save_time_key}: {old_time_str} -> {current_time_str}")
        else:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_time_key] = current_time_str
        
    # auto save path
    if auto_save_path:
        auto_save_call_path = os.getcwd()
        auto_save_cfg_path = cfg_filename

        if not auto_save_path_key in cfg:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_path_key] = {}

        # auto save call path
        if auto_save_call_path_key in cfg[auto_save_path_key]:
            old_call_path = cfg[auto_save_path_key][auto_save_call_path_key]
            if old_call_path != auto_save_call_path:
                cfg[auto_save_path_key][auto_save_call_path_key] = auto_save_call_path
                if not silent_mode:
                    print(f"{YELLOW}change{RESET}{auto_save_call_path_key}: {old_call_path} -> {auto_save_call_path}")
        else:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_path_key][auto_save_call_path_key] = auto_save_call_path

        # auto save cfg path
        if auto_save_cfg_path_key in cfg[auto_save_path_key]:
            old_cfg_path = cfg[auto_save_path_key][auto_save_cfg_path_key]
            if old_cfg_path != auto_save_cfg_path:
                cfg[auto_save_path_key][auto_save_cfg_path_key] = auto_save_cfg_path
                if not silent_mode:
                    print(f"{YELLOW}change{RESET} {auto_save_cfg_path_key}: {old_cfg_path} -> {auto_save_cfg_path}")
        else:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_path_key][auto_save_cfg_path_key] = auto_save_cfg_path
        
    # save config
    OmegaConf.save(config=cfg, f=cfg_filename)
    if not silent_mode:
        print(f'{GREEN}save config to:{RESET}')
        print(f'File "{cfg_filename}"')
    
    return cfg_filename
