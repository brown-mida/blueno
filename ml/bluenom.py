"""
Script to load the data in the logs directory of gpu1708 up to
Elasticsearch.
"""
import pathlib

import os
from elasticsearch_dsl import connections

from blueno.elasticsearch import (
    insert_or_ignore_filepaths, JOB_INDEX,
    TrainingJob,
)


def bluenom(log_dir: pathlib.Path, gpu1708=False, clean_job_names=False):
    """
    Uploads logs in the directory to bluenom. This will
    only upload logs which have uploaded to Slack.

    This is idempotent so it can be run multiple times.
    :return:
    """

    metrics_file_path: pathlib.Path = None

    # noinspection PyTypeChecker
    for filename in sorted(os.listdir(log_dir)):
        file_path = log_dir / filename

        if filename.endswith('.csv'):
            metrics_file_path = log_dir / filename
        elif filename.endswith('.log'):
            print('indexing {}'.format(filename))
            insert_or_ignore_filepaths(file_path,
                                       metrics_file_path,
                                       gpu1708,
                                       clean_job_names=clean_job_names)
        else:
            print('{} is not a log or CSV file'.format(filename))


if __name__ == '__main__':
    print('resetting job index')
    # Creates a connection to our Airflow instance
    connections.create_connection(hosts=['http://104.196.51.205'])
    if JOB_INDEX.exists():
        JOB_INDEX.delete()
    JOB_INDEX.create()
    TrainingJob.init()
    path = pathlib.Path('/gpfs/main/home/lzhu7/elvo-analysis/logs')
    bluenom(path, gpu1708=True, clean_job_names=True)
