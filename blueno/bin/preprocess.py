from argparse import ArgumentParser
import importlib
import logging

from ..pipeline import start_preprocess


def _check_config(config):
    logging.info('checking that config has all required attributes')
    logging.debug('replace arguments with PARAM_GRID')
    logging.debug('PREPROCESS_ARGS: {}'.format(config.PREPROCESS_ARGS))


def main():
    parser = ArgumentParser()
    parser.add_argument('--config',
                        help='The config module (ex. config_luke)',
                        default='config-1')
    args = parser.parse_args()

    logging.info('using config {}'.format(args.config))
    user_config = importlib.import_module(args.config)
    _check_config(user_config)
    start_preprocess(user_config.PREPROCESS_ARGS)


if __name__ == '__main__':
    main()
