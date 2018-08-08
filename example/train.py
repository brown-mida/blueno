from blueno.pipeline import start_train

from configs.train_params import param_grid

USER = 'Andrew'
NUM_GPUS = 1
GPU_OFFSET = 0
LOG_DIR = '../logs/'
SLACK_TOKEN = None
PARAM_GRID = param_grid


def main():
    start_train(PARAM_GRID, USER, num_gpus=NUM_GPUS, gpu_offset=GPU_OFFSET,
                log_dir=LOG_DIR, slack_token=SLACK_TOKEN)


if __name__ == '__main__':
    main()
