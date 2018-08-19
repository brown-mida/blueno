import itertools
import pathlib
import typing

import keras
import matplotlib
import numpy as np
import os
import pandas as pd
import requests
import sklearn.metrics

matplotlib.use('Agg')  # noqa: E402
from matplotlib import pyplot as plt


# TODO(luke): Split this into a slack and a plotting module.


def slack_report(plot_dir: pathlib.Path,
                 name: str,
                 token: str,
                 report: str = None,
                 channel: str = None):
    """
    Uploads a loss graph, accuacy, and confusion matrix plots in addition
    to useful data about the model to Slack.

    :param x_train: the training data
    :param x_valid: the validation array
    :param y_valid: the validation labels, in the same order as x_valid
    :param model: the trained model
    :param history: the history object returned by training the model
    :param name: the name you want to give the model
    :param params: the parameters of the model to attach to the report
    :param token: your slack API token
    :param id_valid: the ids ordered to correspond with y_valid
    :param chunk: whether or not we're analyzing 3D data
    :param plot_dir: the directory to save the plots in
    :return:
    """
    if channel is None:
        channel = '#model-results'

    loss_path = pathlib.Path(plot_dir) / 'loss.png'
    acc_path = pathlib.Path(plot_dir) / 'acc.png'
    cm_path = pathlib.Path(plot_dir) / 'cm.png'

    plots = sorted(os.listdir(plot_dir))
    plots = [x for x in plots if x not in ['loss.png', 'acc.png', 'cm.png']]

    upload_to_slack(loss_path, f'{name}\n\nLoss', token, channel)
    upload_to_slack(acc_path, f'{name}\n\nAccuracy', token, channel)
    if report is not None:
        upload_to_slack(cm_path, report, token, channel)
    else:
        upload_to_slack(cm_path, f'{name}\n\nConfusion Matrix',
                        token, channel)

    for img in plots:
        img_path = pathlib.Path(plot_dir) / img
        upload_to_slack(img_path, f'{name}\n\n{img}', token, channel)


def _create_all_plots(
        x_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        model: keras.Model,
        history: keras.callbacks.History,
        loss_path: pathlib.Path,
        acc_path: pathlib.Path,
        cm_path: pathlib.Path,
        tn_path: pathlib.Path,
        tp_path: pathlib.Path,
        fn_path: pathlib.Path,
        fp_path: pathlib.Path,
        chunk: bool = False,
        id_valid: np.ndarray = None):
    """DEPRECATED. DELETE.
    """
    save_history(history, loss_path, acc_path)
    # TODO: Refactor this
    if chunk:
        y_valid = np.reshape(y_valid, (len(y_valid), 1))
        report = full_multiclass_report(model,
                                        x_valid,
                                        y_valid,
                                        classes=[0, 1],
                                        cm_path=cm_path,
                                        tp_path=tp_path,
                                        fp_path=fp_path,
                                        tn_path=tn_path,
                                        fn_path=fn_path,
                                        id_valid=id_valid,
                                        chunk=chunk)
    else:
        x_mean = np.array([x_train[:, :, :, 0].mean(),
                           x_train[:, :, :, 1].mean(),
                           x_train[:, :, :, 2].mean()])
        x_std = np.array([x_train[:, :, :, 0].std(),
                          x_train[:, :, :, 1].std(),
                          x_train[:, :, :, 2].std()])
        x_valid_standardized = (x_valid - x_mean) / x_std
        report = full_multiclass_report(model,
                                        x_valid_standardized,
                                        y_valid,
                                        classes=[0, 1],
                                        cm_path=cm_path,
                                        tp_path=tp_path,
                                        fp_path=fp_path,
                                        tn_path=tn_path,
                                        fn_path=fn_path,
                                        id_valid=id_valid,
                                        chunk=chunk)
    return report


def write_to_slack(comment, token):
    """
    Write results to slack.
    """
    channels = 'CBUA09G68'

    r = requests.get(
        'https://slack.com/api/chat.postMessage?' +
        'token={}&channel={}&text={}'.format(token, channels, comment))
    return r


def write_iteration_results(params, result, slack_token,
                            job_name=None, job_date=None,
                            purported_accuracy=None,
                            purported_loss=None,
                            purported_sensitivity=None,
                            final=False, i=0):
    """
    Write iteration results (during validation) to Slack.
    """
    if final:
        text = "-----Final Results-----\n"
    else:
        text = "-----Iteration {}-----\n".format(i + 1)
    text += "Seed: {}\n".format(params.seed)
    text += "Params: {}\n".format(params)
    if (job_name is not None):
        text += 'Job name: {}\n'.format(job_name)
        text += 'Job date: {}\n'.format(job_date)
        text += 'Purported accuracy: {}\n'.format(
            purported_accuracy)
        text += 'Purported loss: {}\n'.format(
            purported_loss)
        text += 'Purported sensitivity: {}\n'.format(
            purported_sensitivity)
    if final:
        text += "\n-----Average Results-----\n"
    else:
        text += "\n-----Results-----\n"
    text += 'Loss: {}\n'.format(result[0])
    text += 'Acc: {}\n'.format(result[1])
    text += 'Sensitivity: {}\n'.format(result[2])
    text += 'Specificity: {}\n'.format(result[3])
    text += 'True Positives: {}\n'.format(result[4])
    text += 'False Negatives: {}\n'.format(result[5])
    write_to_slack(text, slack_token)


def upload_to_slack(filename,
                    comment,
                    token,
                    channels='#model-results'):
    """
    Uploads the file at the given path to the channel.

    :param filename:
    :param comment:
    :param token:
    :param channels:
    :return:
    """
    my_file = {
        'file': (str(filename), open(filename, 'rb'), 'png')
    }

    payload = {
        "filename": str(filename),
        "token": token,
        'initial_comment': comment,
        "channels": channels,
    }

    r = requests.post("https://slack.com/api/files.upload",
                      params=payload,
                      files=my_file)
    return r
