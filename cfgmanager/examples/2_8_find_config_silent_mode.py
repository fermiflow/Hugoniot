# This is an example shows how to find in silent mode.
# Run this example by `python 2_8_find_config_silent_mode.py`

from cfgmanager import find
from omegaconf import OmegaConf

# 1. define the directory to search
directory = './data/'

# 2. define the search_dict, which is used to search the config file
#    find function will find the config.yaml file that has the same key-value pairs as search_dict
search_dict = {'num': 4, 'T': 10000}

# 3. find the config file in silent mode
#    when silent_mode is True, the function will not print the found config files and configs
cfg_filenames, cfgs = find(directory, search_dict, silent_mode=True)

# 4. print the found config files and configs
for cfg_filename, cfg in zip(cfg_filenames, cfgs):
    print(f'Config file: {cfg_filename}')
    print(OmegaConf.to_yaml(cfg))
    print()
