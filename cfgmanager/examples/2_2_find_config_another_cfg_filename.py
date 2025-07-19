# This is a minimal example to show how to find another config filename.
# Run this example by `python 2_2_find_config_another_cfg_filename.py`

from cfgmanager import find

# 1. define the directory to search
directory = './data/'

# 2. define the search_dict, which is used to search the config file
#    find function will find the config.yaml file that has the same key-value pairs as search_dict
search_dict = {'num': 4, 'T': 10000}

# 3. give the config file name
#    note that the default value of cfg_filename is 'config.yaml'
#    `cfgmanager.save` will save the config file with the name of 'config.yaml' by default
#    `cfgmanager.find` will find the config file with the name of 'config.yaml' by default
#    you can give another config file name by `cfg_filename` for `cfgmanager.find` function to find
cfg_filename = 'parameters.yaml'

# 4. find the config file
find(directory, search_dict, cfg_filename=cfg_filename)
