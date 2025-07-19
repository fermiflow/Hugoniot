import os
import git
from datetime import datetime
from omegaconf import DictConfig, OmegaConf, open_dict

from cfgmanager import __version__
from cfgmanager.utils import *

def save(cfg: DictConfig,
         path: str, 
         cfg_save_filename: str = "config.yaml",
         auto_save_path: bool = True,
         auto_save_time: bool = True,
         auto_save_time_format: str = "%Y-%m-%d_%H:%M:%S",
         auto_save_git: bool = True,
         auto_save_cfgmanager_version: bool = True,
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
            auto_save_git: (bool), if save git info to config. Default is True.
            auto_save_cfgmanager_version: (bool), if save cfgmanager version to config. Default is True.
            silent_mode: (bool), if run in silent mode. Default is False.
        OUTPUT:
            cfg_filename: (str), saved config file name.
    """

    # auto save keys
    auto_save_time_key = "auto_save_time"
    auto_save_path_key = "auto_save_path"
    auto_save_call_path_key = "call_path"
    auto_save_cfg_path_key = "cfg_path"
    auto_save_git_key = "auto_save_git"
    auto_save_git_branch_key = "branch"
    auto_save_git_hash_key = "hash"
    auto_save_git_message_key = "message"
    auto_save_cfgmanager_version_key = "auto_save_cfgmanager_version"

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
        
    # auto save git
    if auto_save_git:
        try:
            repo = git.Repo(search_parent_directories=True)
        except Exception:
            print(f"{RED}Error occurred while getting git info: {Exception}{RESET}")

        if not auto_save_git_key in cfg:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_git_key] = {}

        # auto save git branch
        if auto_save_git_branch_key in cfg[auto_save_git_key]:
            old_branch = cfg[auto_save_git_key][auto_save_git_branch_key]
            new_branch = repo.active_branch.name
            if old_branch != new_branch:
                cfg[auto_save_git_key][auto_save_git_branch_key] = new_branch
                if not silent_mode:
                    print(f"{YELLOW}change{RESET} {auto_save_git_branch_key}: {old_branch} -> {new_branch}")
        else:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_git_key][auto_save_git_branch_key] = repo.active_branch.name

        # auto save git hash
        if auto_save_git_hash_key in cfg[auto_save_git_key]:
            old_hash = cfg[auto_save_git_key][auto_save_git_hash_key]
            new_hash = repo.head.commit.hexsha
            if old_hash != new_hash:
                cfg[auto_save_git_key][auto_save_git_hash_key] = new_hash
                if not silent_mode:
                    print(f"{YELLOW}change{RESET} {auto_save_git_hash_key}: {old_hash} -> {new_hash}")
        else:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_git_key][auto_save_git_hash_key] = repo.head.commit.hexsha

        # auto save git message
        if auto_save_git_message_key in cfg[auto_save_git_key]:
            old_message = cfg[auto_save_git_key][auto_save_git_message_key]
            new_message = repo.head.commit.message
            if old_message != new_message:
                cfg[auto_save_git_key][auto_save_git_message_key] = new_message
                if not silent_mode:
                    print(f"{YELLOW}change{RESET} {auto_save_git_message_key}: {old_message} -> {new_message}")
        else:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_git_key][auto_save_git_message_key] = repo.head.commit.message.strip()
        
    # auto save cfgmanager version
    if auto_save_cfgmanager_version:
        cfgmanager_version = __version__
        if auto_save_cfgmanager_version_key in cfg:
            old_cfgmanager_version = cfg[auto_save_cfgmanager_version_key]
            if old_cfgmanager_version != cfgmanager_version:
                cfg[auto_save_cfgmanager_version_key] = cfgmanager_version
                if not silent_mode:
                    print(f"{YELLOW}change{RESET} {auto_save_cfgmanager_version_key}: {old_cfgmanager_version} -> {cfgmanager_version}")
        else:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg[auto_save_cfgmanager_version_key] = cfgmanager_version

    # save config
    OmegaConf.save(config=cfg, f=cfg_filename)
    if not silent_mode:
        print(f'{GREEN}save config to:{RESET}')
        print(f'File "{cfg_filename}"')
    
    return cfg_filename
