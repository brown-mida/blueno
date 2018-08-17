from blueno.pipeline import start_train

from configs.train_params_local import param_grid_local

USER = 'Andrew'
GPUS = ['0']
LOG_DIR = '../logs/'
SLACK_TOKEN = None
AIRFLOW_ADDRESS = None


def run_local_param_grid():
    start_train(param_grid_local, USER, gpus=GPUS,
                log_dir=LOG_DIR, slack_token=SLACK_TOKEN,
                airflow_address=AIRFLOW_ADDRESS)


if __name__ == '__main__':
    run_local_param_grid()
