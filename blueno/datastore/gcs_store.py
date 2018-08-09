import os

from google.cloud import storage

from . import DataStore


class GcsStore(DataStore):
    """
    A datastore located in Google Cloud Storage.
    """

    def __init__(self, credentials, bucket_name):
        self.client = storage.Client.from_service_account_json(
            credentials
        )
        os.system(
            'gcloud auth activate-service-account --key-file=' +
            credentials
        )
        self.bucket_name = bucket_name

    def sync_with_datastore(self, path, local_path):
        """
        Syncs the files of a local path to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        os.makedirs(local_path, exist_ok=True)
        exit = os.system(
            'gsutil -m rsync -r -d gs://{}/{} {}'.format(self.bucket_name,
                                                         path, local_path))
        if exit != 0:
            os.rmdir(local_path)
        return exit == 0

    def fetch_from_datastore(self, path, local_path):
        """
        Fetches files of a local path to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(path)
        blob.download_to_filename(local_path)

    def push_to_datastore(self, path, local_path):
        """
        Pushes a dataset (e.g. a preprocessed one) to the datastore.

        :params path: The path within the datastore to store
        :params local_path: The local path where the data is located
        :return: True if successful, False otherwise
        """
        bucket = self.client.get_bucket(self.bucket_name)
        blob = bucket.blob(path)
        blob.upload_from_filename(local_path)

    def push_folder_to_datastore(self, path, local_path):
        """
        Pushes a dataset (e.g. a processed one) to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        exit = os.system(
            'gsutil -m cp -r {} gs://{}/{}'.format(local_path,
                                                   self.bucket_name,
                                                   path))
        return exit == 0
