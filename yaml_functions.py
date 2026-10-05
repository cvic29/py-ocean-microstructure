'''
CV 2026/10/05 : functions associated to yaml parameters files 

contact : clement.vic@ifremer.fr 
''' 
import yaml
from types import SimpleNamespace
from ruamel.yaml import YAML

def load_param(fname="parameters.yaml"):
    with open(fname) as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)

def add_to_yaml(fname, new_params):
    """
    Append new parameters to an existing YAML file
    without losing comments/formatting.

    Parameters
    ----------
    fname : str
        Path to YAML file
    new_params : dict
        Dictionary of new parameters to add
    """
    yaml = YAML()
    yaml.preserve_quotes = True

    # Load existing file (comments preserved)
    with open(fname) as f:
        data = yaml.load(f)

    # Add/overwrite keys
    for k, v in new_params.items():
        data[k] = v

    # Dump back (comments intact, new keys at the end)
    with open(fname, "w") as f:
        yaml.dump(data, f)

