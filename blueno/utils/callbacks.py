"""Custom metrics, callbacks, and plots.
"""

import keras
import numpy as np
import sklearn.metrics


class AucCallback(keras.callbacks.Callback):

    def __init__(self,
                 x_valid_standardized: np.ndarray,
                 y_valid: np.ndarray,
                 job_logger):
        super().__init__()
        self.x_valid_standardized = x_valid_standardized
        self.y_valid = y_valid
        self.job_logger = job_logger

    def on_epoch_end(self, epoch: int, logs=None):
        y_pred = self.model.predict(self.x_valid_standardized)
        score = sklearn.metrics.roc_auc_score(self.y_valid, y_pred)
        self.job_logger.info(f'val_auc: {score}')


class CustomReduceLR(keras.callbacks.ReduceLROnPlateau):
    """
    A LR reduction callback that will not lower the lr before
    epoch 25.

    """

    def __init__(self,
                 x_valid_standardized: np.ndarray,
                 y_valid: np.ndarray,
                 job_logger):
        super().__init__(monitor='val_acc',
                         factor=0.1,
                         verbose=1)

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 25:
            super(CustomReduceLR, self).on_batch_end(epoch, logs)


def create_callbacks(x_train: np.ndarray, y_train: np.ndarray,
                     x_valid: np.ndarray, y_valid: np.ndarray,
                     job_logger,
                     user_defined_callbacks=None,
                     early_stopping: bool = True,
                     csv_file: str = None,
                     model_file: str = None,
                     normalize=True):
    """
    Instantiates a list of callbacks to be fed into the model.

    If csv_file is not None, a CSV logger is instantiated.
    If model_file is not None, a model checkpointer is instantiated.
    If early_stopping is true, an early stopping callback is created.

    The AUC callback is always created.

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param early_stopping: bool flag for whether to do early stopping
        on val_acc
    :param csv_file: the filepath to save the CSV results to
    :param model_file: the filepath to save the models to
    :param normalize: whether or not to normalize the x data
    :return:
    """
    callbacks = []

    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_acc',
            verbose=1,
            patience=10
        ))

    if csv_file:
        callbacks.append(keras.callbacks.CSVLogger(csv_file, append=True))

    if model_file:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            model_file,
            monitor='val_acc',
            verbose=1,
            save_best_only=True
        ))

    if normalize:
        x_mean = np.array([x_train[:, :, :, i].mean()
                           for i in range(x_train.shape[-1])])
        x_std = np.array([x_train[:, :, :, i].std()
                          for i in range(x_train.shape[-1])])
        x_valid_standardized = (x_valid - x_mean) / x_std
    else:
        x_valid_standardized = x_valid

    if user_defined_callbacks is not None:
        for cb in user_defined_callbacks:
            callbacks.append(cb(x_valid_standardized, y_valid, job_logger))

    return callbacks
