import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

seed_job = 2022


class TileDataset(Dataset):
    def __init__(self,
                 metadata: str,
                 transform=None,
                 augmentation=None,
                 resize=512):

        df_metadata = pd.read_csv(metadata)
        self.metadata = df_metadata

        self.resize = resize

        if len(df_metadata.index) == 0:
            raise RuntimeError(f'No Input Files in {metadata}!')
        logging.info(f'Creating Dataset: {self.metadata.shape} examples...')

    def __len__(self):
        return len(self.metadata.index)

    @classmethod
    def preprocess(cls, img, mask, resize):
        #         w, h = img.size
        #         w_sized, h_sized = 224, 224
        img = img.resize((resize, resize))
        img_ndarray = np.asarray(img)

        if img_ndarray.ndim == 2 and not mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
            img_ndarray = img_ndarray / 255
        elif not mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def loader(cls, filename: str):
        return Image.open(filename).convert("RGB")

    def __getitem__(self, idx: int):
        label = self.metadata.label[idx]
        tile = self.metadata.path[idx]
        wsi = self.metadata.wsi[idx]
        resize = self.resize

        img = self.loader(tile)
        ann = "dummy"  # {"stroma": 0, "normal": 1, "pannet": 2}[label]

        img = self.preprocess(img, mask=False, resize=resize)

        return {'image': torch.as_tensor(img.copy()).float().contiguous(),
                'ann': ann, 'name_img': tile, 'wsi': wsi}
