"""
This script allows you to train and evaluate models by specifying a
dictionary of data source and hyperparameter values (in args).

Instead of worrying about how to evaluate/debug models, you can
instead just preprocess data, define a model and have this code
take care of the rest.

To use this script, one would configure the arguments at the bottom
of the file and then run the script using nohup on a GPU machine.
You would then come back in a while to see your results (on Slack).

The proposed workflow is:
- define processed data (filepaths)
- define data generators (methods taking in x_train, y_train and other params)
    - define data augmentation as well
- define models to train (methods taking in generators and other params)
    - also define hyperparameters to optimize as well

After doing such, run the script and see your model results on Slack
in a few minutes.

The script assumes that:
- you have ssh access to one of our configured cloud GPU
- you are able to get processed data onto that computer
- you are familiar with Python and the terminal
"""

import datetime
import logging
import multiprocessing
import pathlib
import time
from typing import List, Union

import keras
import numpy as np
import os
from google.auth.exceptions import DefaultCredentialsError
from sklearn import model_selection

from ..utils.gcs import (
    upload_model_to_gcs,
    upload_gcs_plots,
    equal_array_counts,
    download_to_gpu1708
)
from ..utils.logger import (
    configure_job_logger,
    configure_parent_logger
)
from ..utils.metrics import (
    true_positives,
    false_negatives,
    sensitivity,
    specificity
)
from ..utils.elasticsearch import (
    insert_or_ignore_filepaths,
    create_connection
)
from ..utils.slack import slack_report
from ..utils.preprocessing import prepare_data
from ..utils.callbacks import create_callbacks
from ..types import ParamConfig, ParamGrid

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def start_job(x_train: np.ndarray,
              y_train: np.ndarray,
              x_valid: np.ndarray,
              y_valid: np.ndarray,
              job_name: str,
              username: str,
              params: ParamConfig,
              slack_token: str = None,
              airflow_address: str = None,
              log_dir: str = None,
              plot_dir=None,
              id_valid: np.ndarray = None) -> None:
    """
    Builds, fits, and evaluates a model.

    If slack_token is not none, uploads an image.

    For advanced users it is recommended that you input your own job
    function and attach desired loggers.

    :param x_train:
    :param y_train: the training labels, must be a 2D array
    :param x_valid:
    :param y_valid: the validation labels, must be  a 2D array
    :param job_name:
    :param username:
    :param slack_token: the slack token
    :param params: the parameters specified
    :param log_dir:
    :param plot_dir: the directory to save plots to, defaults to /tmp/plots-
    :param id_valid: the patient ids ordered to correspond with y_valid
    :return:
    """
    num_classes = y_train.shape[1]
    created_at = datetime.datetime.utcnow().isoformat()

    if plot_dir is None:
        gpu = os.environ["CUDA_VISIBLE_DEVICES"]
        plot_dir = pathlib.Path('tmp') / f'plots-{gpu}'

    # Configure the job to log all output to a specific file
    csv_filepath = None
    log_filepath = None
    if log_dir:
        if '/' in job_name:
            raise ValueError("Job name cannot contain '/' character")
        log_filepath = str(pathlib.Path(log_dir) /
                           f'{job_name}-{created_at}.log')
        assert log_filepath.startswith(log_dir)
        csv_filepath = log_filepath[:-3] + 'csv'
        configure_job_logger(log_filepath)

    # This must be the first lines in the jo log, do not change
    logging.info(f'using params:\n{params}')
    logging.info(f'author: {username}')

    logging.debug(f'in start_job,'
                  f' using gpu {os.environ["CUDA_VISIBLE_DEVICES"]}')

    logging.info('preparing data and model for training')

    model_params = params.model
    generator_params = params.generator

    train_gen, valid_gen = generator_params.generator_callable(
        x_train, y_train,
        x_valid, y_valid,
        params.batch_size,
        **generator_params.__dict__)

    logging.debug(f'num_classes is: {num_classes}')

    # Construct the uncompiled model
    model: keras.Model
    model = model_params.model_callable(input_shape=x_train.shape[1:],
                                        num_classes=num_classes,
                                        **model_params.__dict__)

    logging.debug(
        'using default metrics: acc, sensitivity, specificity, tp, fn')
    metrics = ['acc', sensitivity, specificity, true_positives,
               false_negatives]

    model.compile(optimizer=model_params.optimizer,
                  loss=model_params.loss,
                  metrics=metrics)

    model_filepath = '/tmp/{}.hdf5'.format(os.environ['CUDA_VISIBLE_DEVICES'])
    logging.debug('model_filepath: {}'.format(model_filepath))
    callbacks = create_callbacks(x_train, y_train, x_valid, y_valid,
                                 early_stopping=params.early_stopping,
                                 reduce_lr=params.reduce_lr,
                                 csv_file=csv_filepath,
                                 model_file=model_filepath)
    logging.info('training model')
    history = model.fit_generator(train_gen,
                                  epochs=params.max_epochs,
                                  validation_data=valid_gen,
                                  verbose=2,
                                  callbacks=callbacks)

    try:
        upload_gcs_plots(x_train, x_valid, y_valid, model, history,
                         job_name, created_at, plot_dir=plot_dir,
                         id_valid=id_valid)
    except DefaultCredentialsError as e:
        logging.warning(e)

    if slack_token:
        logging.info('generating slack report')
        slack_report(x_train, x_valid, y_valid, model, history,
                     job_name, params, slack_token, plot_dir=plot_dir,
                     id_valid=id_valid)
    else:
        logging.info('no slack token found, not generating report')

    # acc_i = model.metrics_names.index('acc')
    # TODO(luke): Document this change, originally we only upload good models,
    # now we upload all models to GCS
    # if model.evaluate_generator(valid_gen)[acc_i] >= 0.8:
    upload_model_to_gcs(job_name, created_at, model_filepath)

    end_time = datetime.datetime.utcnow().isoformat()
    # Do not change, this generates the ended at ES field
    logging.info(f'end time: {end_time}')

    # Upload logs to Kibana
    if log_dir is not None and airflow_address is not None:
        # Creates a connection to our Airflow instance
        # We don't need to remove since the process ends
        create_connection(hosts=[airflow_address])
        insert_or_ignore_filepaths(
            pathlib.Path(log_filepath),
            pathlib.Path(csv_filepath),
        )


def hyperoptimize(hyperparams: Union[ParamGrid, List[ParamConfig]],
                  username: str,
                  slack_token: str = None,
                  airflow_address: str = None,
                  num_gpus=1,
                  gpu_offset=0,
                  log_dir: str = None) -> None:
    """
    Runs training jobs on input hyperparameter grid.

    :param hyperparams: a dictionary of parameters. See blueno/types for
    a specification
    :param username: your name
    :param slack_token: a slack token for uploading to GitHub
    :param num_gpus: the number of gpus you will use
    :param gpu_offset: your gpu offset
    :param log_dir: the directory you will too. This directory should already
    exist
    :return:
    """
    if isinstance(hyperparams, ParamGrid):
        param_list = model_selection.ParameterGrid(hyperparams.__dict__)
    else:
        param_list = hyperparams

    logging.info(
        'optimizing grid with {} configurations'.format(len(param_list)))

    gpu_index = 0
    processes = []
    for params in param_list:
        if isinstance(params, dict):
            params = ParamConfig(**params)

        check_data_in_sync(params)

        # This is where we'd run preprocessing. To run in a reasonable amount
        # of time, the raw data must be cached in-memory.
        arrays = prepare_data(params, train_test_val=False)
        x_train, x_valid, y_train, y_valid, id_train, id_valid = arrays

        # Start the model training job
        # Run in a separate process to avoid memory issues
        # Note how this depends on offset
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_index + gpu_offset}'

        if params.job_fn is None:
            job_fn = start_job
        else:
            job_fn = params.job_fn

        logging.debug('using job fn {}'.format(job_fn))

        # Uses the parent of the data_dir to name the job,
        # which may not work for all data formats.
        if params.job_name:
            job_name = params.job_name
        else:
            job_name = str(pathlib.Path(params.data.data_dir).parent.name)
        job_name += f'_{y_train.shape[1]}-classes'

        process = multiprocessing.Process(
            target=job_fn,
            args=(x_train, y_train,
                  x_valid, y_valid),
            kwargs={
                'params': params,
                'job_name': job_name,
                'username': username,
                'slack_token': slack_token,
                'airflow_address': airflow_address,
                'log_dir': log_dir,
                'id_valid': id_valid,
            }
        )
        gpu_index += 1
        gpu_index %= num_gpus

        logging.debug(f'gpu_index is now {gpu_index + gpu_offset}')
        process.start()
        processes.append(process)
        if gpu_index == 0:
            logging.info(f'all gpus used, calling join on processes:'
                         f' {processes}')
        p: multiprocessing.Process
        for p in processes:
            p.join()
        processes = []
        time.sleep(60)


def check_data_in_sync(params: ParamConfig):
    """
    Checks that the data is in-sync with google cloud.

    This is so we can reproduce and ensemble the arrays.


    TODO(luke): Refactor
    If the data doesn't exist and we're on gpu1708, an attempt to download
    the data from GCS will be made, so the web trainer works.

    This also assumes that gcs_url/arrays contains the arrays.

    :param params:
    :return:
    """
    data_dir = pathlib.Path(params.data.data_dir)
    gcs_url = params.data.gcs_url

    if gcs_url is None:
        logging.warning('No GCS url found, will not check for syncing')
        return

    if gcs_url.endswith('/'):
        array_url = gcs_url + 'arrays'
    else:
        array_url = gcs_url + '/arrays'

    try:
        is_equal = equal_array_counts(data_dir, array_url)
    except FileNotFoundError:
        # TODO(luke): Refactor this
        if os.uname().nodename == 'gpu1708':
            logging.info(f'data on GCS does not exist locally,'
                         f' downloading data to {data_dir}')
            download_to_gpu1708(array_url, data_dir, folder=True)
            # TODO(luke): Allow web users to generate labels
            default_label_url = \
                'gs://elvos/processed/processed-lower/labels.csv'
            labels_path = params.data.labels_path
            download_to_gpu1708(default_label_url, labels_path)
    except DefaultCredentialsError as e:
        logging.warning(e)
        logging.warning('Will not check GCS for syncing')
    else:
        if not is_equal:
            raise ValueError(f'{data_dir} and {array_url} have a different'
                             f' number of files')


def start_train(param_grid, user, num_gpus=1, gpu_offset=0,
                log_dir='logs/', slack_token=None, airflow_address=None,
                configure_logger=True):
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if configure_logger:
        parent_log_file = pathlib.Path(
            log_dir) / 'results-{}.txt'.format(
            datetime.datetime.utcnow().isoformat()
        )
        configure_parent_logger(parent_log_file)

    logging.info('Checking param grid...')
    if isinstance(param_grid, ParamGrid):
        param_grid = param_grid
    elif isinstance(param_grid, list):
        param_grid = param_grid
    elif isinstance(param_grid, dict):
        logging.warning('creating param grid from dictionary, it is'
                        'recommended that you define your config'
                        'with ParamConfig')
        param_grid = ParamGrid(**param_grid)
    else:
        raise ValueError('param_grid must be a ParamGrid,'
                         ' list, or dict')
    hyperoptimize(param_grid, user, slack_token, airflow_address,
                  num_gpus, gpu_offset, log_dir)


def start_train_from_config(config):
    start_train(config.PARAM_GRID, config.USER,
                num_gpus=config.NUM_GPUS,
                gpu_offset=config.GPU_OFFSET,
                log_dir=config.LOG_DIR,
                slack_token=config.SLACK_TOKEN)
