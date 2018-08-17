import numpy as np

from blueno.types import PreprocessConfig
from blueno.datastore import LocalStore
from blueno.pipeline import start_preprocess_from_config


def expand_dims(arr):
    return np.expand_dims(arr, axis=2)


PREPROCESS_ARGS = PreprocessConfig(
    datastore=LocalStore(),
    arrays_dir='../data/mnist_data/arrays/',
    arrays_compressed=False,
    labels_dir='../data/mnist_data/labels.csv',
    labels_index_col='ID',
    labels_value_col='Label',
    processed_dir='../data/processed_mnist/',
    filter_func=None,
    process_func=expand_dims
)


def main():
    start_preprocess_from_config(PREPROCESS_ARGS)


if __name__ == '__main__':
    main()
