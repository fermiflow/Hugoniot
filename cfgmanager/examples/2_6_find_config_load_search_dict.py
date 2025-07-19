# This is an example shows how to load search_dict from a yaml file.
# Run this example by `python 2_6_find_config_load_search_dict.py`

from cfgmanager import find
from omegaconf import OmegaConf

# 1. define the directory to search
directory = './data/'

# 2. load search_dict from a yaml file
search_dict = OmegaConf.load('./conf/search_dict.yaml')

# 3. find the config file
find(directory, search_dict)
