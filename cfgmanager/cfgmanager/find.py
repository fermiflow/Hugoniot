import re
import os
import pandas as pd
from typing import Tuple, Union
from omegaconf import OmegaConf, DictConfig

from cfgmanager.utils import *

def compare_config_dict(config_dict, search_dict):
    """
        Compare config dict and search dict.
        INPUT:
            config_dict: config dict. (dict)
            search_dict: search dict. (dict)
        OUTPUT:
            True if config_dict contains search_dict, otherwise False. (bool)
        Note:
            search_dict only contains key-value pairs that need to be compared.
            config_dict usually contains more key-value pairs than search_dict.
    """
    for key in search_dict:
        if key not in config_dict:
            return False
        if isinstance(config_dict[key], DictConfig) and isinstance(search_dict[key], dict):
            if not compare_config_dict(config_dict[key], search_dict[key]):
                return False
        else:
            if isinstance(search_dict[key], str):
                if not re.match(search_dict[key], config_dict[key]):
                    return False
            else:
                if config_dict[key] != search_dict[key]:
                    return False
    return True

def search_config(directory: str, 
                  search_dict: Union[dict, DictConfig],
                  cfg_filename: Union[str, list] = 'config.yaml',
                  all_yaml: bool = False,
    ) -> Tuple[list, list]:
    """
        Search config.yaml files in directory with search_dict.
        INPUT:
            directory: directory to search. (str)
            search_dict: search dict. (dict or omegaconf.DictConfig)
            cfg_filename: config file name. Default is 'config.yaml'. (str or list)
                          if cfg_filename is a list, search all config file names in the list.
            all_yaml: if search all yaml files. Default is False. (bool)
                      when all_yaml is True, cfg_filename is ignored.
        OUTPUT:
            matched_cfg_filenames: matched config.yaml file names. (list of str)
            matched_cfgs: matched configs. (list of omegaconf.DictConfig)
    """
    cfg_filenames = []
    if all_yaml:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.yaml'):
                    cfg_filenames.append(os.path.join(root, file))
    else:
        if isinstance(cfg_filename, str):
            for root, dirs, files in os.walk(directory):
                if cfg_filename in files:
                    cfg_filenames.append(os.path.join(root, cfg_filename))
        elif isinstance(cfg_filename, list):
            for root, dirs, files in os.walk(directory):
                for cfg_fname in cfg_filename:
                    if cfg_fname in files:
                        cfg_filenames.append(os.path.join(root, cfg_fname))

    matched_cfg_filenames = []
    matched_cfgs = []
    for cfg_filename in cfg_filenames:
        cfg = OmegaConf.load(cfg_filename)
        if compare_config_dict(cfg, search_dict):
            matched_cfg_filenames.append(cfg_filename)
            matched_cfgs.append(cfg)
    return matched_cfg_filenames, matched_cfgs

def flatten_dict(nested_dict, parent_key='', sep='_'):
    """
        Flatten nested dict.
    """
    items = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items

def extract_keys(nested_dict: dict, keys_to_extract: dict) -> dict:
    """
        Extract keys from nested dict.
        INPUT:
            nested_dict: nested dict. (dict)
            keys_to_extract: keys to extract. (dict)
    """
    small_dict = {}
    for key in keys_to_extract.keys():
        if key in nested_dict:
            if isinstance(keys_to_extract[key], dict):
                small_dict[key] = extract_keys(nested_dict[key], keys_to_extract[key])
            else:
                small_dict[key] = nested_dict[key]
    return small_dict

def print_configs(cfgs: list, print_dict: dict = None) -> None:
    """
        Print configs.
        Input:
            cfgs: list of omegaconf.DictConfig
            print_dict: dict to print, only print_dict.keys() matter. (dict)
    """
    flat_cfg_dicts = []
    for cfg in cfgs:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if print_dict is not None:
            cfg_dict = extract_keys(cfg_dict, print_dict)
        flat_cfg_dict = flatten_dict(cfg_dict)
        flat_cfg_dicts.append(flat_cfg_dict)

    df = pd.DataFrame.from_dict(flat_cfg_dicts)
    print(df)

def find(directory: str, 
         search_dict: Union[dict, DictConfig], 
         cfg_filename: Union[str, list] = 'config.yaml',
         all_yaml: bool = False,
         print_dict: Union[dict, DictConfig] = None,
         print_ignore_auto_save: bool = True,
         silent_mode: bool = False,
    ) -> Tuple[list, list]:
    """
        Find config files in directory with search_dict.
        INPUT:
            directory: directory to search. (str)
            search_dict: search dict. (dict)
            cfg_filename: config file name. Default is 'config.yaml'. (str or list)
                          if cfg_filename is a list, search all config file names in the list.
            all_yaml: if search all yaml files. Default is False. (bool)
                        when all_yaml is True, cfg_filename is ignored.
            print_dict: dict to print, only print_dict.keys() matter. (dict)
            print_ignore_auto_save: if print ignore auto save keys when print_dict is None. Default is True. (bool)
            silent_mode: if run in silent mode. Default is False. (bool)
        OUTPUT:
            matched_cfg_filenames: matched config.yaml file names. (list of str)
            matched_cfgs: matched configs. (list of omegaconf.DictConfig)
    """
    directory = os.path.abspath(directory)
    matched_cfg_filenames, matched_cfgs = search_config(directory, search_dict, cfg_filename, all_yaml)

    if not silent_mode:

        print(f"\n{GREEN}Find {len(matched_cfg_filenames)} matched config files:{RESET}")
        for matched_cfg_filename in matched_cfg_filenames:
            print(f'File "{matched_cfg_filename}"')

        if len(matched_cfgs) > 0:
            if print_dict is None:
                print_dict = matched_cfgs[0]
                if isinstance(print_dict, DictConfig):
                    print_dict = OmegaConf.to_container(print_dict, resolve=True)
                if print_ignore_auto_save:
                    pattern = re.compile(r'^auto_save_')
                    print_dict = {k: v for k, v in print_dict.items() if not pattern.match(k)}
            else:
                if isinstance(print_dict, DictConfig):
                    print_dict = OmegaConf.to_container(print_dict, resolve=True)
            print(f"\n{GREEN}Brief matched configs:{RESET}")
            print_configs(matched_cfgs, print_dict)

    return matched_cfg_filenames, matched_cfgs
