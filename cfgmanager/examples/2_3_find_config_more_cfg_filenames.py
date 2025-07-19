# This is an example shows how to find two or more config file names in the directory.
# Run this example by `python 2_3_find_config_more_cfg_filenames.py`

from cfgmanager import find

# 1. define the directory to search
directory = './data/'

# 2. define the search_dict, which is used to search the config file
#    find function will find the config.yaml file that has the same key-value pairs as search_dict
search_dict = {'num': 4, 'T': 10000}

# 3. give the config file names in a list
#    the `cfg_filename` parameter of `cfgmanager.find` can also be a list of strings
cfg_filenames = ['config.yaml', 'parameters.yaml']

# 4. find the config file
find(directory, search_dict, cfg_filename=cfg_filenames)
