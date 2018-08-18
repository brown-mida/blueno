"""
Define new parameters here.

By being explict about parameters, it will be easier to share
information around the team.

All generator and model functions should take in **kwargs as an argument.
"""

import typing
from dataclasses import dataclass

import numpy as np
from sklearn import model_selection

from ..datastore import DataStore


@dataclass
class PreprocessConfig:
    datastore: DataStore
    arrays_dir: str
    arrays_compressed: bool
    labels_dir: str
    labels_index_col: str
    labels_value_col: str
    processed_dir: str
    filter_func: typing.Callable[[str, np.ndarray], bool]
    process_func: typing.Callable[[np.ndarray], np.ndarray]


@dataclass
class DataConfig:
    datastore: DataStore
    data_dir: str
    index_col: str
    value_col: str
    results_dir: str


# This class is experimental and subject to a lot of change.
# When a clear solution for appending preprocessing to the
# config grid arises, this will likely be replaced.
@dataclass
class LukePipelineConfig(DataConfig):
    """
    Defines a configuration to preprocess raw numpy data.
    """
    pipeline_callable: typing.Callable
    height_offset: int
    mip_thickness: int
    pixel_value_range: typing.Sequence


@dataclass
class ModelConfig:
    # Should return a model and must take in **kwargs as an argument
    model_callable: typing.Callable
    optimizer: typing.Callable
    loss: typing.Callable


@dataclass
class GeneratorConfig:
    # Should return a tuple containing x_train, y_train, x_test, y_test
    # Must take in **kwargs as an argument
    generator_callable: typing.Callable


@dataclass
class EvalConfig:
    """
    Config data structure required to evaluate a given model.
    """
    model: ModelConfig
    model_weights: str
    data: DataConfig
    val_split: float
    seed: int


# Keep the below two classes in sync

@dataclass
class ParamConfig:
    data: DataConfig
    generator: GeneratorConfig
    model: ModelConfig
    batch_size: int
    seed: int
    val_split: float
    callbacks: typing.Sequence[typing.Callable]

    max_epochs: int = 100
    early_stopping: bool = True

    job_fn: typing.Callable = None
    job_name: str = None


@dataclass
class ParamGrid:
    data: typing.Sequence[DataConfig]
    generator: typing.Sequence[GeneratorConfig]
    model: typing.Sequence[ModelConfig]
    batch_size: typing.Sequence[int]
    seed: typing.Sequence[int]
    val_split: typing.Sequence[int]
    callbacks: typing.Sequence[typing.Sequence[typing.Callable]]

    max_epochs: typing.Tuple[int] = (100,)
    early_stopping: typing.Sequence[bool] = (True,)

    job_fn: typing.Sequence[typing.Callable] = None
    job_name: str = None

    def __init__(self, **kwargs):
        for attr in kwargs:
            if attr not in ParamGrid.__dataclass_fields__:
                raise ValueError(
                    '{} is not an attribute of ParamGrid'.format(attr))

        # With a grid you can only define one of DataConfig and
        # PipelineConfig, use a list of ParamConfigs instead if you want
        # more configurability
        if not isinstance(kwargs['data'], DataConfig):
            if 'pipeline_callable' in kwargs:
                data = tuple(LukePipelineConfig(**d) for d in kwargs['data'])
                self.data = data
            else:
                data = tuple(DataConfig(**d) for d in kwargs['data'])
                self.data = data

        if not isinstance(kwargs['generator'], GeneratorConfig):
            generators = tuple(
                GeneratorConfig(**gen) for gen in kwargs['generator'])
            self.generator = generators

        if not isinstance(kwargs['model'], ModelConfig):
            models = tuple(ModelConfig(**mod) for mod in kwargs['model'])
            self.model = models

        self.seed = kwargs['seed']
        self.val_split = kwargs['val_split']
        self.batch_size = kwargs['batch_size']

        if 'job_fn' in kwargs:
            self.job_fn = kwargs['job_fn']


if set(ParamConfig.__dataclass_fields__.keys()) \
        != set(ParamGrid.__dataclass_fields__.keys()):
    raise ValueError(
        'ParamConfig and ParamGrid do not have the same properties')


def create_param_grid(config_type, params):
    config_list = list(model_selection.ParameterGrid(params))
    return [config_type(**m) for m in config_list]
