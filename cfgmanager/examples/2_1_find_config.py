# This is a minimal example to show how to save the config file.
# Run this example by `python 2_1_find_config.py`

from cfgmanager import find

# 1. define the directory to search
directory = './data/'

# 2. define the search_dict, which is used to search the config file
#    find function will find the config.yaml file that has the same key-value pairs as search_dict
search_dict = {'num': 4, 'T': 10000}

# 3. find the config file
find(directory, search_dict)
