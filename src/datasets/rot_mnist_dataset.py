# torch
import torch
import torch.optim
import torch.utils.data
from torch.utils.data import random_split, DataLoader, TensorDataset

# built-in
import numpy as np


# Taken from https://github.com/tscohen/gconv_experiments/tree/master/gconv_experiments (Cohen & Welling, 2016)
def preprocess_mnist_data(train_data, test_data, train_labels, test_labels):
    train_mean = np.mean(train_data)  # compute mean over all pixels
    train_data -= train_mean
    test_data -= train_mean
    train_std = np.std(train_data)
    train_data /= train_std
    test_data /= train_std
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    # Return preprocessed dataset
    return train_data, test_data, train_labels, test_labels


def get_dataset(batch_size, num_workers, val_split=0.1):
    # Load dataset
    train_set = np.load('../data/train_all.npz')  # adjust path as needed
    test_set = np.load('../data/test.npz')

    train_data, train_labels = train_set['data'], train_set['labels']
    test_data, test_labels = test_set['data'], test_set['labels']

    train_data, test_data, train_labels, test_labels = preprocess_mnist_data(
        train_data, test_data, train_labels, test_labels
    )

    train_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_labels_tensor = torch.from_numpy(train_labels).type(torch.LongTensor)
    test_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_labels_tensor = torch.from_numpy(test_labels).type(torch.LongTensor)

    full_train_dataset = TensorDataset(train_tensor, train_labels_tensor)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = TensorDataset(test_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
