from dataclasses import dataclass
from typing import Union, Tuple

from keras import optimizers, losses

from blueno.types import (
    DataConfig,
    ModelConfig,
    GeneratorConfig,
    ParamConfig,
    create_param_grid
)
from blueno.models.luke import resnet
from blueno.generators.luke import standard_generators
from blueno.datastore import GcsStore


@dataclass
class ResnetModelConfig(ModelConfig):
    dropout_rate1: int = 0.8
    dropout_rate2: int = 0.8
    freeze: bool = False


@dataclass
class ResnetGeneratorConfig(GeneratorConfig):
    rotation_range: int = 30
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0
    zoom_range: Union[float, Tuple[float, float]] = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = False


model_list = create_param_grid(ResnetModelConfig, {
    'model_callable': [resnet],
    'optimizer': [
        optimizers.Adam(lr=1e-5),
    ],
    'loss': [
        losses.categorical_crossentropy,
    ],
    'dropout_rate1': [0.8],
    'dropout_rate2': [0.8],
    'freeze': [False],
})

gen_list = create_param_grid(ResnetGeneratorConfig, {
    'generator_callable': [standard_generators],
    'rotation_range': [30]
})

store = GcsStore('../credentials/client_secret.json', 'elvos')

data_list = create_param_grid(DataConfig, {
    'datastore': [store],
    'arrays_dir': ['processed/processed-lower-nbv/arrays'],
    'labels_dir': ['processed/processed-lower-nbv/labels.csv'],
    'index_col': ['Anon ID'],
    'value_col': ['occlusion_exists'],
    'local_tmp_dir': ['../data/processed-lower-nbv/']
})

param_grid = create_param_grid(ParamConfig, {
    'data': data_list,
    'generator': gen_list,
    'model': model_list,
    'batch_size': [8],
    'seed': [0, 1, 2, 3, 4, 5],
    'val_split': [0.1, 0.2, 0.3],
    'reduce_lr': [True, False],
    'early_stopping': [False],
    'max_epochs': [60],
})
