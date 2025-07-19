# This is an example shows how to find all config file in the directory.
# Run this example by `python 2_4_find_config_all_yaml.py`

from cfgmanager import find

# 1. define the directory to search
directory = './data/'

# 2. define the search_dict, which is used to search the config file
#    find function will find the config.yaml file that has the same key-value pairs as search_dict
search_dict = {'num': 4, 'flow': {'depth': 6}}

# 3. find the config file
#    when `all_yaml` is True, the function will find all the yaml files in the directory
find(directory, search_dict, all_yaml=True)
