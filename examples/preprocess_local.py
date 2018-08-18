import numpy as np

from blueno.types import PreprocessConfig
from blueno.datastore import LocalStore
from blueno.pipeline import start_preprocess_from_config


def make_three_channels(arr):
    return np.tile(arr, (1, 1, 3))


PREPROCESS_ARGS = PreprocessConfig(
    datastore=LocalStore(),
    arrays_dir='../data/mnist_data/arrays/',
    arrays_compressed=False,
    labels_dir='../data/mnist_data/labels.csv',
    labels_index_col='ID',
    labels_value_col='Label',
    processed_dir='../data/processed_mnist/',
    filter_func=None,
    process_func=make_three_channels,
    plot_all_preview=False
)


def main():
    start_preprocess_from_config(PREPROCESS_ARGS)


if __name__ == '__main__':
    main()
