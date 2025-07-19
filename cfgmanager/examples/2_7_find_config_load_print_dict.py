# This is an example shows how to load print_dict from a yaml file.
# Run this example by `python 2_7_find_config_load_print_dict.py`

from cfgmanager import find
from omegaconf import OmegaConf

# 1. define the directory to search
directory = './data/'

# 2. load search_dict from a yaml file
search_dict = OmegaConf.load('./conf/search_dict.yaml')

# 3. load print_dict from a yaml file
print_dict = OmegaConf.load('./conf/print_dict.yaml')

# 4. find the config file
find(directory, search_dict, print_dict=print_dict)
