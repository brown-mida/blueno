from dataclasses import dataclass

from keras import optimizers, losses

from blueno.types import (
    DataConfig,
    ModelConfig,
    GeneratorConfig,
    ParamConfig,
    create_param_grid
)
from blueno.models.basic_cnn import basic_cnn_model
from blueno.generators.luke import standard_generators
from blueno.utils.callbacks import CustomReduceLR
from blueno.datastore import LocalStore


@dataclass
class MnistModelConfig(ModelConfig):
    filter_size1: int
    filter_size2: int
    hidden_size: int


@dataclass
class MnistGeneratorConfig(GeneratorConfig):
    width_shift_range: float
    height_shift_range: float


model_list = create_param_grid(MnistModelConfig, {
    'model_callable': [basic_cnn_model],
    'optimizer': [
        optimizers.Adam(lr=1e-5),
    ],
    'loss': [
        losses.categorical_crossentropy,
    ],
    'filter_size1': [32, 64],
    'filter_size2': [64, 128],
    'hidden_size': [300, 500, 700],
})

gen_list = create_param_grid(MnistGeneratorConfig, {
    'generator_callable': [standard_generators],
    'width_shift_range': [0, 0.05],
    'height_shift_range': [0, 0.05]
})

data_list = create_param_grid(DataConfig, {
    'datastore': [LocalStore()],
    'data_dir': ['../data/processed_mnist/'],
    'index_col': ['ID'],
    'value_col': ['Label'],
    'results_dir': ['../data/processed_mnist/results/']
})

param_grid_local = create_param_grid(ParamConfig, {
    'data': data_list,
    'generator': gen_list,
    'model': model_list,
    'batch_size': [8],
    'seed': [0, 1, 2],
    'val_split': [0.1, 0.2, 0.3],
    'callbacks': [[CustomReduceLR]],
    'early_stopping': [False],
    'max_epochs': [2],
})
