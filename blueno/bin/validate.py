import os
import sys
import logging
import argparse
import importlib

from ..pipeline.bluenov import start_validation, get_validation_params
from ..types import ParamConfig


def parse_args(args):
    """
    Parse arguments for this script.
    """
    parser = argparse.ArgumentParser(description='Evaluation script.')
    subparsers = parser.add_subparsers(
        help='Arguments for specific evaluation types.',
        dest='eval_type'
    )
    subparsers.required = True

    param_parser = subparsers.add_parser('param-list')
    param_parser.add_argument(
        'param-list-config',
        help=('Path to config file. This file must have a list of '
              'ParamConfig objects stored in EVAL_PARAM_LIST.')
    )

    kibana_parser = subparsers.add_parser('kibana')
    kibana_parser.add_argument('--address',
                               help='Address to access Kibana.',
                               default='http://104.196.51.205')
    kibana_parser.add_argument('--lower',
                               help='Lower bound of best_val_acc to search.',
                               default='0.85')
    kibana_parser.add_argument('--upper',
                               help='Upper bound of best_val_acc to search.',
                               default='0.93')

    parser.add_argument(
        '--gpu',
        help=('Ids of the GPU to use (as reported by nvidia-smi). '
              'Separated by comma, no spaces. e.g. 0,1'),
        default=None
    )
    parser.add_argument(
        '--log-dir',
        help=('Location to store logs.'),
        default='../logs/'
    )

    parser.add_argument(
        '--data-dir',
        help=('Location to store temporary files.'),
        default='../tmp/'
    )

    parser.add_argument(
        '--config',
        help=('Configuration file, if you want to specify GPU and '
              'log directories there.'),
        default=None
    )
    parser.add_argument(
        '--num-iterations',
        help=('Number of times to test a parameter config. '
              'Evaluation results are averaged.'),
        default=1
    )
    parser.add_argument(
        '--slack-token',
        help=('Slack token, to upload results to slack.'),
        default=None
    )

    parser.add_argument(
        '--no-early-stopping',
        help=('Saves best model after running max epochs, '
              'instead of early stopping'),
        action='store_true'
    )

    return parser.parse_args(args)


def check_user_config(config):
    """
    Check param-list-config to see if it is valid.
    """
    logging.info('Checking that user config has all required attributes')
    logging.info('LOG_DIR: {}'.format(config.LOG_DIR))
    logging.info('DATA_DIR: {}'.format(config.DATA_DIR))
    logging.info('gpus: {}'.format(config.gpus))
    logging.info('num_iterations: {}'.format(config.num_iterations))
    logging.info('SLACK_TOKEN: {}'.format(config.SLACK_TOKEN))
    logging.info('no_early_stopping: {}'.format(config.no_early_stopping))
    for attr in ['LOG_DIR', 'DATA_DIR', 'gpus', 'num_iterations',
                 'SLACK_TOKEN', 'no_early_stopping']:
        if not hasattr(config, attr):
            raise AttributeError('User config file is missing {}'.format(attr))


def check_config(config):
    """
    Check param-list-config to see if it is valid.
    """
    logging.info('Checking that config has all required attributes')
    logging.info('EVAL_PARAM_LIST: {}'.format(config.EVAL_PARAM_LIST))
    if (not (isinstance(config.EVAL_PARAM_LIST, list)) or
       not (isinstance(config.EVAL_PARAM_LIST[0], ParamConfig))):
        raise ValueError('EVAL_PARAM_LIST must be a list of ParamConfig')


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Set config if exists
    if args.config is not None:
        user_config = importlib.import_module(args.config)
        check_user_config(user_config)

        args.log_dir = user_config.LOG_DIR
        args.data_dir = user_config.DATA_DIR
        args.gpu = user_config.gpus
        args.num_iterations = user_config.num_iterations
        args.slack_token = user_config.SLACK_TOKEN
        args.no_early_stopping = user_config.no_early_stopping

    # Choose GPU to use
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpus = args.gpu.split(',')
    else:
        gpus = ['0']

    # Set configurations
    num_iterations = args.num_iterations
    slack_token = args.slack_token
    no_early_stopping = args.no_early_stopping

    if args.eval_type == 'kibana':
        # Fetch params list from Kibana to evaluate
        models = get_validation_params(args.address, args.lower,
                                       args.upper, args.data_dir)
        start_validation(models, num_iterations=num_iterations, gpus=gpus,
                         slack_token=slack_token,
                         no_early_stopping=no_early_stopping,
                         address=args.address, log_dir=args.log_dir,
                         data_dir=args.data_dir)

    else:
        # Manual evaluation of a list of ParamConfig
        logging.info('Using config {}'.format(args.param_list_config))
        param_list_config = importlib.import_module(args.param_list_config)
        check_config(param_list_config)

        params = param_list_config.EVAL_PARAM_LIST
        start_validation(params, num_iterations=num_iterations, gpus=gpus,
                         slack_token=slack_token,
                         no_early_stopping=no_early_stopping,
                         address=args.address, log_dir=args.log_dir,
                         data_dir=args.data_dir)


if __name__ == '__main__':
    main()
