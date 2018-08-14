"""
This script allows you to preprocess a dataset by simply specifying
the directories (of the data and labels), and functions to filter
and preprocess the data. Visualizations of the data will also be
saved to the post-processed directory.

The idea is that we want to abstract away the burden behind
loading and saving data, as well as getting rid of
odd data (e.g. data missing labels).
"""

import os
import pathlib
import typing
import logging
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shutil import rmtree

from ..utils.io import (
    load_compressed_arrays,
    load_arrays,
    load_raw_labels
)
from ..utils.slack import plot_images
from ..utils.preprocessing import clean_data
from ..types import PreprocessConfig


def filter_data(arrays: typing.Dict[str, np.ndarray],
                labels: pd.DataFrame,
                condition: typing.Callable[[str, np.ndarray], bool]):
    filtered_arrays = {id_: arr for id_, arr in arrays.items()
                       if arr.shape[0] != 1}  # Remove the bad array
    filtered_labels = labels.loc[filtered_arrays.keys()]

    filtered_arrays = filtered_arrays.copy()
    if condition is not None:
        filtered_ids = [id_ for id_, arr in filtered_arrays.items()
                        if condition(id_, arr)]
        for id_ in list(filtered_arrays):
            if id_ not in filtered_ids:
                del filtered_arrays[id_]
        filtered_labels = labels.loc[filtered_ids]

    assert len(filtered_arrays) == len(filtered_labels)
    return filtered_arrays, filtered_labels


def _process_arrays(arrays: typing.Dict[str, np.ndarray],
                    process_func: typing.Callable[[np.ndarray], np.ndarray]
                    ) -> typing.Dict[str, np.ndarray]:
    processed = {}
    for id_, arr in arrays.items():
        try:
            processed[id_] = process_func(arr)
        except AssertionError as e:
            print('Patient id {} could not be processed: {}'.format(id_, e))
    return processed


def process_data(arrays, labels, process_func):
    if process_func is not None:
        arrays = _process_arrays(arrays, process_func)
        labels = labels.loc[arrays.keys()]
    assert len(arrays) == len(labels)
    return arrays, labels


def save_plots(arrays, labels, dirpath: str, value_col: str):
    os.mkdir(dirpath)
    num_plots = (len(arrays) + 19) // 20
    for i in range(num_plots):
        print(f'saving plot number {i}')
        plot_images(arrays, labels, value_col, num_cols=5, offset=20 * i)
        plt.savefig(f'{dirpath}/{20 * i}-{20 * i + 19}')
        plt.close()


def save_data(arrays: typing.Dict[str, np.ndarray],
              labels: pd.DataFrame,
              dirpath: str,
              value_col: str,
              with_plots=True):
    """
    Saves the arrays and labels in the given dirpath.

    :param arrays:
    :param labels:
    :param dirpath:
    :param with_plots:
    :return:
    """
    # noinspection PyTypeChecker
    os.makedirs(pathlib.Path(dirpath) / 'arrays', exist_ok=True)
    for id_, arr in arrays.items():
        print(f'saving {id_}')
        # noinspection PyTypeChecker
        np.save(pathlib.Path(dirpath) / 'arrays' / f'{id_}.npy', arr)
    labels.to_csv(pathlib.Path(dirpath) / 'labels.csv')
    plots_dir = str(pathlib.Path(dirpath) / 'plots')
    if with_plots:
        try:
            save_plots(arrays, labels, plots_dir, value_col)
        except Exception as e:
            print('Warning: Could not save plot. {}'.format(e))


def start_preprocess(datastore, arrays_dir, labels_dir, labels_index_col,
                     labels_value_col, processed_dir, filter_func,
                     process_func, compressed=True):
    created_at = datetime.datetime.utcnow().isoformat()
    local_tmp_dir = f'/tmp/blueno/{created_at}'

    datastore.sync_dataset(arrays_dir, labels_dir, local_tmp_dir)
    local_arrays_dir = os.path.join(local_tmp_dir, 'arrays/')
    local_labels_dir = os.path.join(local_tmp_dir, 'labels.csv')
    local_processed_dir = os.path.join(local_tmp_dir, 'processed/')
    os.mkdir(local_processed_dir)

    if compressed:
        raw_arrays = load_compressed_arrays(local_arrays_dir)
    else:
        raw_arrays = load_arrays(local_arrays_dir)
    raw_labels = load_raw_labels(local_labels_dir, labels_index_col)
    filtered_arrays, filtered_labels = filter_data(raw_arrays,
                                                   raw_labels,
                                                   filter_func)
    processed_arrays, processed_labels = process_data(filtered_arrays,
                                                      filtered_labels,
                                                      process_func)
    cleaned_arrays, cleaned_labels = clean_data(processed_arrays,
                                                processed_labels)
    save_data(cleaned_arrays, cleaned_labels, local_processed_dir,
              labels_value_col)
    datastore.push_folder_to_datastore(processed_dir,
                                       local_processed_dir)
    rmtree(local_tmp_dir)


def start_preprocess_from_config(args: typing.Union[PreprocessConfig, dict]):
    """
    Runs a preprocessing job with the args.

    :param args: The config dictionary for the processing job
    :return:
    """
    if isinstance(args, PreprocessConfig):
        args = args
    elif isinstance(args, dict):
        logging.warning('A dictionary was specified for arguments. It is'
                        'recommended that you define your arguments'
                        'with PreprocessConfig.')
        args = PreprocessConfig(**args)
    else:
        raise ValueError('args must be a PreprocessConfig '
                         'or dict')

    start_preprocess(args.datastore, args.arrays_dir, args.labels_dir,
                     args.labels_index_col, args.labels_value_col,
                     args.processed_dir, args.filter_func,
                     args.process_func, args.arrays_compressed)
