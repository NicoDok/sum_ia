import numpy as np
from addict import Dict

from config import DATASET


def read_dataset(filepath: str = DATASET) -> Dict:
    x = np.load(filepath)
    return Dict(
        train_x=x['train_x'],
        train_y=x['train_y'],
        train_label=x['train_label'],
        validation_x=x['validation_x'],
        validation_y=x['validation_y'],
        validation_label=x['validation_label'],
        test_x=x['test_x'],
        test_y=x['test_y'],
        test_label=x['test_label'],
    )