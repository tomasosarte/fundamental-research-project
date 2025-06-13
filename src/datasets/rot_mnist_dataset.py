import os
import urllib.request
import zipfile
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def download_mnist_rotation(datadir='../data'):
    os.makedirs(datadir, exist_ok=True)
    zip_path = os.path.join(datadir, 'mnist_rotation_new.zip')

    # Skip if already present
    if all(os.path.exists(os.path.join(datadir, f)) for f in ['train_all.npz', 'test.npz']):
        print("Rotated MNIST already prepared.")
        return

    print("Downloading rotated MNIST dataset...")
    url = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting contents...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(datadir)
    os.remove(zip_path)

    print("Converting .amat files to .npz...")

    train_fn = os.path.join(datadir, 'mnist_all_rotation_normalized_float_train_valid.amat')
    test_fn = os.path.join(datadir, 'mnist_all_rotation_normalized_float_test.amat')

    train_val = np.loadtxt(train_fn)
    test = np.loadtxt(test_fn)

    train_val_data = train_val[:, :-1].reshape(-1, 1, 28, 28)
    train_val_labels = train_val[:, -1]

    test_data = test[:, :-1].reshape(-1, 1, 28, 28)
    test_labels = test[:, -1]

    np.savez(os.path.join(datadir, 'train_all.npz'), data=train_val_data, labels=train_val_labels)
    np.savez(os.path.join(datadir, 'train.npz'), data=train_val_data[:10000], labels=train_val_labels[:10000])
    np.savez(os.path.join(datadir, 'valid.npz'), data=train_val_data[10000:], labels=train_val_labels[10000:])
    np.savez(os.path.join(datadir, 'test.npz'), data=test_data, labels=test_labels)

    print("Done. Saved to:", datadir)

def preprocess_mnist_data(train_data, test_data, train_labels, test_labels):
    train_mean = np.mean(train_data)
    train_data -= train_mean
    test_data -= train_mean
    train_std = np.std(train_data)
    train_data /= train_std
    test_data /= train_std
    return train_data.astype(np.float32), test_data.astype(np.float32), \
           train_labels.astype(np.int64), test_labels.astype(np.int64)

def get_dataset(batch_size=128, num_workers=2, datadir='../data'):
    download_mnist_rotation('../data')

    # Load
    train_npz = np.load(os.path.join(datadir, 'train.npz'))
    valid_npz = np.load(os.path.join(datadir, 'valid.npz'))
    test_npz = np.load(os.path.join(datadir, 'test.npz'))

    train_data, train_labels = train_npz['data'], train_npz['labels']
    valid_data, valid_labels = valid_npz['data'], valid_npz['labels']
    test_data, test_labels = test_npz['data'], test_npz['labels']

    # Normalize all sets with same stats
    train_data, test_data, train_labels, test_labels = preprocess_mnist_data(
        train_data, test_data, train_labels, test_labels
    )
    _, valid_data, _, valid_labels = preprocess_mnist_data(train_data, valid_data, train_labels, valid_labels)

    # To TensorDatasets
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    val_dataset = TensorDataset(torch.from_numpy(valid_data), torch.from_numpy(valid_labels))
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader