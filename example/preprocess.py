from blueno.types import PreprocessConfig
from blueno.datastore import GcsStore
from blueno.pipeline import start_preprocess_from_config

store = GcsStore('../credentials/client_secret.json', 'elvos')


PREPROCESS_ARGS = PreprocessConfig(
    datastore=store,
    arrays_dir='test_dataset/',
    labels_dir='labels.csv',
    labels_index_col='ID',
    labels_value_col='Label',
    processed_dir='processed_dataset/',
    local_tmp_dir='../tmp/dataset/',
    filter_func=None,
    process_func=None
)


def main():
    start_preprocess_from_config(PREPROCESS_ARGS)


if __name__ == '__main__':
    main()
