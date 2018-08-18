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


def save_plots(x_valid: np.ndarray,
               y_valid: np.ndarray,
               model: keras.models.Model,
               history: keras.callbacks.History,
               plot_dir: str,
               num_classes: int,
               id_valid: np.ndarray):
    """
    Uploads a loss graph, accuracy, and confusion matrix plots in addition
    to useful data about the model to gcs.

    Raises an error if credentials are not set.

    Saves to gs://elvos-public/plots/{job_name}-{created_at}/

    :param x_train:
    :param x_valid:
    :param y_valid:
    :param model:
    :param history:
    :param job_name:
    :param params:
    :param token:
    :param id_valid:
    :param chunk:
    :param plot_dir:
    :return:
    """
    os.makedirs(plot_dir, exist_ok=True)
    save_history(history, plot_dir)
    report = full_multiclass_report(model,
                                    x_valid,
                                    y_valid,
                                    num_classes=num_classes,
                                    plot_dir=plot_dir,
                                    id_valid=id_valid)
    return report


def save_history(history: keras.callbacks.History,
                 plot_dir: pathlib.Path):
    """
    Saves plots of the loss/acc over epochs in the given paths.

    :param history:
    :param loss_path:
    :param acc_path:
    :return:
    """
    loss_path = os.path.join(plot_dir, 'loss.png')
    acc_path = os.path.join(plot_dir, 'acc.png')

    loss_list = [s for s in history.history.keys() if
                 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if
                     'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if
                'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if
                    'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    plt.figure()
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(
                     str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(
                     str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_path)

    plt.figure()
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(
                     format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(
                     format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_path)


def full_multiclass_report(model: keras.models.Model,
                           x: np.ndarray,
                           y_true: np.ndarray,
                           num_classes: int,
                           plot_dir: pathlib.Path,
                           id_valid: np.ndarray):
    """
    Builds a report containing the following:
        - accuracy
        - AUC
        - classification report
        - confusion matrix
        - 7/31/2018 metrics

    The output is the report as a string.

    The report also generates a confusion matrix plot and tp/fp examples.

    :param model:
    :param x:
    :param y_true:
    :param classes:
    :param id_valid
    :param chunk
    :return:
    """
    # TODO(luke): Split this into separate functions.
    y_proba = model.predict(x, batch_size=8)
    assert y_true.shape == y_proba.shape

    if y_proba.shape[-1] == 1:
        y_pred = (y_proba > 0.5).astype('int32')
    else:
        y_pred = y_proba.argmax(axis=1)
        y_true = y_true.argmax(axis=1)

    assert y_pred.shape == y_true.shape, \
        f'y_pred.shape: {y_pred.shape} must equal y_true.shape: {y_true.shape}'

    comment = "Accuracy: " + str(
        sklearn.metrics.accuracy_score(y_true, y_pred))
    comment += '\n'

    if num_classes <= 2:
        # Calculate AUC. Assuming 0 is the negative label.
        y_true_binary = y_true > 0
        y_pred_binary = y_pred > 0
        score = sklearn.metrics.roc_auc_score(y_true_binary,
                                              y_pred_binary)

        # Do not change the line below, it affects reporting._extract_auc
        comment += f'AUC: {score}\n'
        comment += f'Assuming {0} is the negative label'
        comment += '\n\n'

    comment += "Classification Report\n"
    comment += sklearn.metrics.classification_report(y_true, y_pred, digits=5)

    cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    comment += '\n'
    comment += str(cnf_matrix)
    save_confusion_matrix(cnf_matrix, plot_dir=plot_dir)

    if num_classes <= 2:
        # Additional metrics for binary classification.
        try:
            tn, fp, fn, tp = cnf_matrix.ravel()
            comment += f'\n\nAdditional statistics:\n'
            sensitivity = tp / (tp + fn)
            comment += f'Sensitivity: {sensitivity}\n'
            specificity = tn / (tn + fp)
            comment += f'Specificity: {tn / (tn + fp)}\n'
            comment += f'Precision: {tp / (tp + fp)}\n'
            total_acc = (tp + tn) / (tp + tn + fp + fn)
            random_acc = (((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp))
                          / (tp + tn + fp + fn) ** 2)
            comment += f'\n\nNamed statistics:\n'
            kappa = (total_acc - random_acc) / (1 - random_acc)
            comment += f'Cohen\'s Kappa: {kappa}\n'
            youdens = sensitivity - (1 - specificity)
            comment += f'Youden\'s index: {youdens}\n'

            comment += f'\n\nOther sklearn statistics:\n'
            log_loss = sklearn.metrics.classification.log_loss(y_true, y_pred)
            comment += f'Log loss: {log_loss}\n'
            comment += f'F-1: {sklearn.metrics.f1_score(y_true, y_pred)}\n'
        except ValueError as e:
            comment += '\nCould not add additional statistics (tp, fp, etc.)'
            comment += str(e)

    save_misclassification_plots(x,
                                 y_true,
                                 y_pred,
                                 num_classes=num_classes,
                                 plot_dir=plot_dir,
                                 id_valid=id_valid)
    return comment


def save_confusion_matrix(cm,
                          plot_dir: pathlib.Path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints, plots, and saves the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_path = os.path.join(plot_dir, 'cm.png')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, range(len(cm)), rotation=45)
    plt.yticks(tick_marks, range(len(cm)))

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(cm_path)


def save_misclassification_plots(x,
                                 y_true,
                                 y_pred,
                                 num_classes: int,
                                 plot_dir: pathlib.Path,
                                 id_valid: np.ndarray):
    """Saves the 4 true/false positive/negative plots.

    The y inputs must be binary and 1 dimensional.
    """
    # Address binary edge case
    if num_classes < 2:
        num_classes = 2

    for i in range(num_classes):
        correct_path = os.path.join(plot_dir, f'{i}-correct.png')
        incorrect_path = os.path.join(plot_dir, f'{i}-wrong.png')
        mask_correct = np.logical_and(y_pred == i, y_true == i)
        mask_incorrect = np.logical_and(y_pred == i, y_true != i)

        x_filtered_correct = x[mask_correct]
        x_filtered_correct_labels = id_valid[mask_correct]
        x_filtered_incorrect = x[mask_incorrect]
        x_filtered_incorrect_labels = id_valid[mask_incorrect]

        plot_misclassification(x_filtered_correct,
                               y_true[mask_correct],
                               y_pred[mask_correct],
                               x_filtered_correct_labels)
        plt.savefig(correct_path)
        plt.close()

        plot_misclassification(x_filtered_incorrect,
                               y_true[mask_incorrect],
                               y_pred[mask_incorrect],
                               x_filtered_incorrect_labels)
        plt.savefig(incorrect_path)
        plt.close()


# def __save_misclassification_plots(x_valid,
#                                    y_true,
#                                    y_pred,
#                                    plot_dir: pathlib.Path,
#                                    id_valid: np.ndarray = None,
#                                    chunk=False):
#     """Saves the 4 true/false positive/negative plots.

#     The y inputs must be binary and 1 dimensional.
#     """
#     assert len(x_valid) == len(y_true)
#     if y_true.max() > 1 or y_pred.max() > 1:
#         raise ValueError('y_true/y_pred should be binary 0/1')

#     plot_name_dict = {
#         (0, 0): tn_path,
#         (1, 1): tp_path,
#         (0, 1): fp_path,
#         (1, 0): fn_path,
#     }

#     for i in (0, 1):
#         for j in (0, 1):
#             mask = np.logical_and(y_true == i, y_pred == j)
#             x_filtered = np.array([x_valid[i] for i, truth in enumerate(mask)
#                                    if truth])

#             if id_valid is None:
#                 ids_filtered = None
#             else:
#                 ids_filtered = id_valid[mask]

#             plot_misclassification(x_filtered,
#                                    y_true[mask],
#                                    y_pred[mask],
#                                    ids=ids_filtered,
#                                    chunk=chunk)
#             plt.savefig(plot_name_dict[(i, j)])


def plot_misclassification(x,
                           y_true,
                           y_pred,
                           ids: np.ndarray,
                           num_cols=5,
                           limit=20,
                           offset=0):
    """
    Plots the figures with labels and predictions.

    :param x:
    :param y_true:
    :param y_pred:
    :param num_cols:
    :param limit:
    :param offset:
    :param ids:
    :param chunk:
    :return:
    """
    num_rows = (min(len(x), limit) + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(10, 10))
    for i, arr in enumerate(x):
        if i < offset:
            continue
        if i >= offset + limit:
            break
        plot_num = i - offset + 1
        ax = fig.add_subplot(num_rows, num_cols, plot_num)
        ax.set_title(f'ID: {ids[i]}')
        ax.set_xlabel(f'y_true: {y_true[i]} y_pred: {y_pred[i]}')
        plt.imshow(arr)
    fig.tight_layout()
    plt.plot()


def plot_images(data: typing.Dict[str, np.ndarray],
                labels: pd.DataFrame,
                value_col='occlusion_exists',
                num_cols=5,
                limit=20,
                offset=0):
    """
    Plots limit images in a single plot.

    :param data:
    :param labels:
    :param num_cols:
    :param limit: the number of images to plot
    :param offset:
    :return:
    """
    # Ceiling function of len(data) / num_cols
    num_rows = (min(len(data), limit) + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(10, 10))
    for i, index_id in enumerate(data):
        if i < offset:
            continue
        if i >= offset + limit:
            break
        plot_num = i - offset + 1
        ax = fig.add_subplot(num_rows, num_cols, plot_num)
        ax.set_title(f'Index: {index_id[:4]}...')
        label = labels.loc[index_id][value_col]
        ax.set_xlabel(f'label: {label}')
        plt.imshow(data[index_id])
    fig.tight_layout()
    plt.plot()
