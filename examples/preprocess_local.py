from blueno.types import PreprocessConfig
from blueno.datastore import LocalStore
from blueno.pipeline import start_preprocess_from_config

store = LocalStore()

PREPROCESS_ARGS = PreprocessConfig(
    datastore=store,
    arrays_dir='../data/sample_dataset/arrays/',
    arrays_compressed=False,
    labels_dir='../data/sample_dataset/labels.csv',
    labels_index_col='ID',
    labels_value_col='Label',
    processed_dir='../data/processed_dataset/',
    local_tmp_dir='/tmp/datasets/',
    filter_func=None,
    process_func=None
)


def main():
    start_preprocess_from_config(PREPROCESS_ARGS)


if __name__ == '__main__':
    main()
