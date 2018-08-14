import os
import filecmp
import logging
from shutil import copytree, rmtree, copy

from . import DataStore


class LocalStore(DataStore):
    """
    A datastore located in your local storage.
    """

    def fetch_folder_from_datastore(self, path, local_path):
        """
        Syncs the files of a local path to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        try:
            if os.path.isdir(local_path):
                rmtree(local_path)
            copytree(path, local_path)
            return True
        except Exception:
            return False

    def fetch_from_datastore(self, path, local_path):
        """
        Fetches files of a local path to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        copy(path, local_path)

    def push_to_datastore(self, path, local_path):
        """
        Pushes a dataset (e.g. a preprocessed one) to the datastore.

        :params path: The path within the datastore to store
        :params local_path: The local path where the data is located
        :return: True if successful, False otherwise
        """
        copy(local_path, path)

    def push_folder_to_datastore(self, path, local_path):
        """
        Pushes a dataset (e.g. a processed one) to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        try:
            if os.path.isdir(path):
                logging.error('A path in the datastore with the same '
                              'name exists.')
                raise ValueError('Duplicate path')
            copytree(local_path, path)
            return True
        except Exception:
            return False

    def dataset_is_equal(self, array_path, label_path, local_path):
        """
        Checks if the dataset in the datastore is the same as the one
        in the local path.

        :params array_path: Path of the data in the datastore
        :params label_path: Path of the labels in the datastore
        :params local_path: The local path to store the fetched data
        :return: True if data is equal, False otherwise
        """
        if not os.path.isdir(local_path):
            return False

        local_array_path = os.path.join(local_path, 'arrays/')
        local_label_path = os.path.join(local_path, 'labels.csv')

        # Check file equality
        local_list = set(os.listdir(local_array_path))
        datastore_list = set(os.listdir(array_path))
        files_same = local_list == datastore_list

        # Check label equality
        labels_same = filecmp.cmp(label_path, local_label_path)

        return files_same and labels_same
