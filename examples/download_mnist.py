"""
Forked from https://github.com/hsjeong5/MNIST-for-Numpy

Downloads a readily-formatted MNIST dataset for use for
the example scripts in the examples/ folder. The dataset will
be stored in data/mnist_dataset/.
"""

import os
from urllib import request
import gzip
import csv

import numpy as np

filename = [
    "train-images-idx3-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]


def process_mnist(data_dir):
    # Download MNIST Data
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print(f"Downloading {name}...")
        request.urlretrieve(base_url+name, f'/tmp/{name}')
    print("Download complete.")

    # Save data to proper format
    os.makedirs(data_dir, exist_ok=True)
    data = np.zeros((0, 28, 28))
    for name in filename[:2]:
        with gzip.open(f'/tmp/{name}', 'rb') as f:
            data_tmp = np.frombuffer(
                f.read(), np.uint8, offset=16
            ).reshape(-1, 28, 28)
            data = np.concatenate((data, data_tmp))
    for i in range(data.shape[0]):
        image = data[i, :, :]
        np.save(f'{data_dir}/{i}.npy', image)
        print(f"Saving data... {round(i / data.shape[0] * 100, 1)}%",
              end="\r")

    # Save labels to proper format
    data = np.zeros((0,))
    for name in filename[-2:]:
        with gzip.open(f'/tmp/{name}', 'rb') as f:
            data_tmp = np.frombuffer(
                f.read(), np.uint8, offset=8
            )
            data = np.concatenate((data, data_tmp))
    print("\nSaving labels...")
    with open(f'{data_dir}/labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID', 'Label'])
        for i in range(data.shape[0]):
            writer.writerow([i, int(data[i])])

    # Cleanup
    for name in filename:
        os.remove(f'/tmp/{name}')
    print("Save complete.")


if __name__ == '__main__':
    process_mnist('../data/mnist_data/')
