import os


class DataStore(object):

    def fetch_folder_from_datastore(self, path, local_path):
        """
        Syncs the files of a local path to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        raise NotImplementedError('fetch_from_datastore not implemented')

    def fetch_from_datastore(self, path, local_path):
        """
        Fetches files of a local path to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        """
        raise NotImplementedError('fetch_from_datastore not implemented')

    def push_to_datastore(self, path, local_path):
        """
        Pushes a file or local path to the datastore.

        :params path: The path within the datastore to store
        :params local_path: The local path where the data is located
        :return: True if successful, False otherwise
        """
        raise NotImplementedError('push_to_datastore not implemented')

    def push_folder_to_datastore(self, path, local_path):
        """
        Pushes a dataset (e.g. a processed one) to the path of the datastore

        :params path: The path within the datastore
        :params local_path: The local path to store the fetched data
        :return: True if successful, False otherwise
        :raise: ValueError if the datastore path exists. We do not want to
            overwrite datastore paths.
        """
        raise NotImplementedError('fetch_from_datastore not implemented')

    def dataset_is_equal(self, array_path, label_path, local_path):
        """
        Checks if the dataset in the datastore is the same as the one
        in the local path.

        :params array_path: Path of the data in the datastore
        :params label_path: Path of the labels in the datastore
        :params local_path: The local path to store the fetched data
        :return: True if data is equal, False otherwise
        """
        raise NotImplementedError('check_if_dataset_equal not implemented')

    def sync_dataset(self, array_path, label_path, local_path):
        """
        Fetches relevant information of a datset and puts them in
        an organized local folder.

        :params array_path: Path of the data in the datastore
        :params label_path: Path of the labels in the datastore
        :params local_path: The local path to store the fetched data
        :return: True if sync succeeds, False otherwise
        """
        os.makedirs(local_path, exist_ok=False)
        local_arrays_dir = os.path.join(local_path, 'arrays/')
        local_labels_dir = os.path.join(local_path, 'labels.csv')
        local_processed_dir = os.path.join(local_path, 'processed/')
        os.mkdir(local_arrays_dir)
        os.mkdir(local_processed_dir)
        fetched_data = self.fetch_folder_from_datastore(array_path,
                                                        local_arrays_dir)
        fetched_labels = self.fetch_from_datastore(label_path,
                                                   local_labels_dir)
        return fetched_data and fetched_labels
