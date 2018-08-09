import os

from shutil import copytree, rmtree, copy

from . import DataStore


class GcsStore(DataStore):
    """
    A datastore located in your local storage.
    """

    def sync_with_datastore(self, path, local_path):
        """
        Syncs the files of a local path to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        try:
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
            rmtree(path)
            os.makedirs(path, exist_ok=True)
            copytree(local_path, path)
            return True
        except Exception:
            return False
