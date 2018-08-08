from argparse import ArgumentParser
import importlib
import logging
import pathlib
import datetime

from ..pipeline import start_train
from ..utils.logger import configure_parent_logger


def check_config(config):
    logging.info('checking that config has all required attributes')
    logging.debug('replace arguments with PARAM_GRID')
    logging.debug('PARAM_GRID: {}'.format(config.PARAM_GRID))
    logging.debug('USER: {}'.format(config.USER))
    logging.debug('GPUS: {}'.format(config.GPUS))
    logging.debug('BLUENO_HOME: {}'.format(config.BLUENO_HOME))
    logging.debug('LOG_DIR: {}'.format(config.LOG_DIR))
    logging.debug('SLACK_TOKEN: {}'.format(config.SLACK_TOKEN))


def main():
    parser = ArgumentParser()
    parser.add_argument('--config',
                        help='The config module (ex. config_luke)',
                        default='config-1')
    args = parser.parse_args()

    logging.info('using config {}'.format(args.config))
    user_config = importlib.import_module(args.config)
    parent_log_file = pathlib.Path(
        user_config.LOG_DIR) / 'results-{}.txt'.format(
        datetime.datetime.utcnow().isoformat()
    )
    configure_parent_logger(parent_log_file)
    check_config(user_config)

    start_train(user_config, user_config.USER,
                slack_token=user_config.SLACK_TOKEN,
                gpus=user_config.GPUS,
                log_dir=user_config.LOG_DIR,
                configure_logger=False)


if __name__ == '__main__':
    main()
