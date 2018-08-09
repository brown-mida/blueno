from blueno.types import PreprocessConfig
from blueno.datastore import GcsStore
from blueno.pipeline import start_preprocess_from_config

store = GcsStore('../../credentials/client_secret.json', 'elvos')


preprocess_config = PreprocessConfig({
    'arrays_dir': 'test_dataset/',
    'labels_dir': 'labels.csv',
    'local_tmp_dir': '../tmp/dataset/',
    'filter_func': None,
    'process_func': None
})

start_preprocess_from_config(preprocess_config)
