import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.p = p
        self.transforms = transforms

        with open(os.path.join(base_folder, 'batches.meta'), 'rb') as f:
            self.label_names = pickle.load(f, encoding='bytes')[b'label_names']

        if train:
            self.X = np.zeros((50000, 3, 32, 32), dtype=np.uint8)
            self.y = np.zeros((50000,), dtype=np.uint8)
            for i in range(1, 6):
                with open(os.path.join(base_folder, f'data_batch_{i}'), 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                self.X[(i-1)*10000:i*10000] = data[b'data'].reshape((10000, 3, 32, 32))
                self.y[(i-1)*10000:i*10000] = data[b'labels']
        else:
            with open(os.path.join(base_folder, 'test_batch'), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            self.X = data[b'data'].reshape((10000, 3, 32, 32))
            self.y = np.array(data[b'labels'])

        self.X = self.X.astype(np.float32) / 255.
        self.y = self.y.astype(np.int64)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X = self.X[index]
        y = self.y[index]

        if self.transforms is not None:
            for tform in self.transforms:
                if not isinstance(index, int):
                    for i in range(len(index)):
                        X[i] = tform(X[i])
                else:
                    X = tform(X)
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
