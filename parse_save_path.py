import json
from pathlib import Path
from collections import OrderedDict
import argparse
import sys

def read_json(file_path):
    file_path = Path(file_path)
    with file_path.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def main(args):
    result = read_json(args.config)
    save_path = f'{result["trainer"]["save_dir"]}/models/{result["name"]}'
    save_path += f"/{result['optimizer']['type']}-lr_{result['optimizer']['args']['lr']}"
    if 'lr_scheduler' in result.keys(): save_path += f"_{result['lr_scheduler']['type']}"
    save_path += f"/{result['arch']['type']}"
    save_path = Path(save_path) 
    sys.exit(save_path)

""" Run """
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=None, type=str,  help='config file path (default: None)')
args = parser.parse_args()

main(args)
