import pathlib
import typing
from typing import Dict

import keras
import numpy as np
import os
import pandas as pd

from . import metrics


def load_arrays(data_dir: str) -> Dict[str, np.ndarray]:
    data_dict = {}
    for filename in os.listdir(data_dir):
        patient_id = filename[:-4]  # remove .npy extension
        data_dict[patient_id] = np.load(pathlib.Path(data_dir) / filename)
    return data_dict


def load_model(model_path: str, compile=True):
    # Need to do this otherwise the model won't load
    keras.metrics.sensitivity = metrics.sensitivity
    keras.metrics.specificity = metrics.specificity
    keras.metrics.true_positives = metrics.true_positives
    keras.metrics.false_negatives = metrics.false_negatives

    model = keras.models.load_model(model_path, compile=compile)
    return model


def load_compressed_arrays(data_dir: str,
                           limit=None) -> typing.Dict[str, np.ndarray]:
    """Loads a directory containing npz files.

    The keys will be the keys of the loaded npz dict.
    """
    data = dict()
    filenames = os.listdir(data_dir)
    if limit:
        filenames = filenames[:limit]
    for filename in filenames:
        print(f'Loading file {filename}')
        d = np.load(pathlib.Path(data_dir) / filename)
        data.update(d)  # merge all_data with d
    return data


def load_raw_labels(labels_dir: str, index_col='Anon ID') -> pd.DataFrame:
    """Loads a directory containing the labels.
    file."""
    df = pd.read_csv(labels_dir, index_col=index_col)
    return df
