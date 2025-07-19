# This is an example shows how to use print_dict to print less but more useful information.
# Run this example by `python 2_5_find_config_print_dict.py`

from cfgmanager import find

# 1. define the directory to search
directory = './data/'

# 2. define the search_dict, which is used to search the config file
#    find function will find the config.yaml file that has the same key-value pairs as search_dict
search_dict = {'num': 4, 'flow': {'depth': 6}}

# 3. defind the print_dict, which gives the information you want to print
#    note that the value of print_dict does not matter, 
#    `cfgmanager.find` will only use the keys in print_dict to print the information
print_dict = {
                'num': None, 
                # 'rs': None,
                'T': None,
                'flow': {
                    # 'steps': None,
                    'depth': None,
                    'h1size': None,
                    # 'h2size': None,
                    },
                'batchsize': None,
                'acc_steps': None,
            }

# 4. find the config file
find(directory, search_dict, print_dict=print_dict)
