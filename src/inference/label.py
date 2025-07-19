from omegaconf import DictConfig

# from parser import parse_filename, natural_sort
def auto_label(cfg: DictConfig, label_dict: dict, sep = '_', label: str = ''):
    """
        Automatically generate label.
    """
    for key in label_dict:
        if label != '' and not label.endswith(sep):
            label = label + sep
        if key in cfg:
            if isinstance(label_dict[key], dict):
                label = auto_label(cfg[key], label_dict[key], sep=sep, label=label)
            else:
                label = label + key + sep + str(cfg[key]) + sep
    if label.endswith(sep):
        label = label[:-1]
    return label

def add_newline_if_long(input_str, max_length=35):
    """
        Add newline if input_str is too long.
    """
    if len(input_str) > max_length:
        return input_str[:max_length] + '\n' + add_newline_if_long(input_str[max_length:])
    else:
        return input_str
