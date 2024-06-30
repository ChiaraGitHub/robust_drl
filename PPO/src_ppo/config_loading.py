import json
import argparse
from typing import Dict

def argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    return args

def update_dict(d: Dict, u: Dict, show_warning=False):
    for k, v in u.items():
        if k not in d and show_warning:
            print(f"Key {k} not found in config!")
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v, show_warning)
        else:
            d[k] = v
    return d

def load_config(args, default_config=''):
    print("Loading config file: {}".format(args.config))
    if default_config != '':
        with open(default_config) as f:
            config = json.load(f)
        if 'config' in vars(args).keys():
            with open(args.config) as f:
                update_dict(config, json.load(f))
    if 'config' in vars(args).keys():
        with open(args.config) as f:
            config = json.load(f)

    return config

