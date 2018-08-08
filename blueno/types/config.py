"""Define new parameters here.

By being explict about parameters, it will be easier to share
information around the team.

All generator and model functions should take in **kwargs as an argument.
"""
import typing
from dataclasses import dataclass

import numpy as np
from sklearn import model_selection


@dataclass
class PreprocessConfig:
    arrays_dir: str
    labels_dir: str
    processed_dir: str
    filter_func: typing.Callable[[str, np.ndarray], bool]
    process_func: typing.Callable[[np.ndarray], np.ndarray]


@dataclass
class DataConfig:
    # A directory containing a list of numpy files with
    # patient ID as their filename.
    # This must end with a '/' for the job name to be configured correctly
    data_dir: str
    # A CSV file generated by saving a pandas DataFrame
    labels_path: str
    index_col: str
    label_col: str
    # The url starting with gs:// that contains the data
    gcs_url: str


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
    rotation_range: int = 30
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0
    zoom_range: typing.Union[float,
                             typing.Tuple[float, float]] = 0.1
    horizontal_flip: bool = True
    vertical_flip: bool = False


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

    max_epochs: int = 100
    early_stopping: bool = True
    reduce_lr: bool = False

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

    max_epochs: typing.Tuple[int] = (100,)
    early_stopping: typing.Sequence[bool] = (True,)
    reduce_lr: typing.Sequence[bool] = (False,)

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