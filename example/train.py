from blueno.pipeline import start_train

from configs.train_params import param_grid

USER = 'Andrew'
GPUS = ['0', '1']
LOG_DIR = '../logs/'
SLACK_TOKEN = None
AIRFLOW_ADDRESS = None
PARAM_GRID = param_grid


def main():
    start_train(PARAM_GRID, USER, gpus=GPUS,
                log_dir=LOG_DIR, slack_token=SLACK_TOKEN,
                airflow_address=AIRFLOW_ADDRESS)


if __name__ == '__main__':
    main()
