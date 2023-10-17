import yaml


def load_config(path, default_path=None):
    """
    Load config file
    Args:
        path:                           (str), path to config file
        default_path:                   (str, optional), whether to use default path.

    Returns:
        cfg:                            (dict), config dict

    """
    # load configuration from file itself
    with open(path, 'r' ) as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # if yes, load this config first as default
    # if no, use the default path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def save_config(cfg, path):
    with open(path, 'w') as fp:
        yaml.dump(cfg, fp)


def update_recursive(dict1, dict2):
    """
    update two config dictionaries recursively
    Args:
        dict1:                          (dict), first dictionary to be updated
        dictw:                          (dict), second dictionary which entries should be used

    Returns:

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
