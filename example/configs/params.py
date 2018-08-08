from dataclasses import dataclass
from typing import Union, Tuple

from keras import optimizers, losses

from blueno.types import (
    DataConfig,
    ModelConfig,
    GeneratorConfig,
    create_param_grid
)
from blueno.models.luke import resnet
from blueno.generators.luke import standard_generators


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
    'dropout_rate1': [0.8],
    'dropout_rate2': [0.8],
    'optimizer': [
        optimizers.Adam(lr=1e-5),
    ],
    'loss': [
        losses.categorical_crossentropy,
    ],
    'freeze': [False],
})

gen_list = create_param_grid(ResnetGeneratorConfig, {
    'generator_callable': [standard_generators],
    'rotation_range': [30]
})
