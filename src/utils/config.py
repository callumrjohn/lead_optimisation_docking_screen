import yaml

def deep_update(orig: dict, new: dict):

    """
    Recursively update a dictionary with another dictionary.
    Can be used to update a config dictionaries

    Args:
        orig (dict): The original dictionary to be updated.
        new (dict): The new dictionary to update with.
    """
    for key, val in new.items():
        if isinstance(val, dict) and key in orig:
            deep_update(orig[key], val)
        else:
            orig[key] = val

def load_config(files):
    
    """
    Load YAML configuration files and merge into a single dictionary.

    Args:
        files (list): List of file paths to YAML configuration files.
    
    Returns:
        dict: Merged configuration dictionary
    """

    config = {}
    for file in files:
        with open(file) as f:
            new_cfg = yaml.safe_load(f)
            deep_update(config, new_cfg)
    return config