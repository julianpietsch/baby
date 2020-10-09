"""Mostly used to access models and model-sets"""
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'models'

def modelsets():
    with open(BASE_DIR / 'modelsets.json', 'r') as fd:
        msets = json.load(fd)
    return msets

# Todo: should probably be removed, but used in Tests
def model_path():
    return MODEL_DIR
