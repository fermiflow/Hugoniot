from cfgmanager import find
from src.inference.utils import *

def find_data(directory, search_dict, print_dict):
    cfg_filenames, _ = find(directory, search_dict, print_dict=print_dict)
    for i in range(len(cfg_filenames)):
        data_filename = cfg_filenames[i].replace('config.yaml', 'data.txt')
        if i == 0:
            print(f"\n{GREEN}plot code:{RESET}")
            print(f'files = ["{data_filename}"]')
        else:        
            print(f'files += ["{data_filename}"]')
