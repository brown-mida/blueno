import os
import filecmp

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

    def fetch_folder_from_datastore(self, path, local_path):
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
        bucket = self.client.get_bucket(self.bucket_name)
        gcs_blobs = bucket.list_blobs(prefix=array_path)
        gcs_list = set()
        for each in gcs_blobs:
            gcs_list.add(each.name.split('/')[-1])
        local_list = set(os.listdir(local_array_path))
        return (gcs_list == local_list and
                filecmp.cmp(label_path, local_label_path))
