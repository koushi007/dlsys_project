from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(label_filename, 'rb') as f:
            self.label_buff = f.read()
            self.num_labels = struct.unpack_from('>2i', self.label_buff)[1]

            labels = struct.unpack_from(str(self.num_labels)+'B', self.label_buff, offset=8)
            self.y = np.array(labels, dtype=np.uint8)

        with gzip.open(image_filename, 'rb') as f:
            self.img_buff = f.read()
            img_header = struct.unpack_from('>4i', self.img_buff)
            self.num_images = img_header[1]
            self.img_rows = img_header[2]
            self.img_cols = img_header[3]
            self.img_pixels = self.img_rows*self.img_cols

            pixels = struct.unpack_from(str(self.num_images*self.img_pixels)+'B', self.img_buff, offset=16)
            self.X = (np.array(pixels, dtype=np.float32)/255).reshape(
                        (self.num_images, self.img_rows, self.img_cols, 1))

        self.transforms = transforms

    def __getitem__(self, index) -> object:
        # X = struct.unpack_from(str(self.img_pixels)+'B', self.img_buff, offset=16+index*self.img_pixels)
        # y = struct.unpack_from('B', self.label_buff, offset=8+index)[0]

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

    def __len__(self) -> int:
        return self.num_images
