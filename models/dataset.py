from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import torch
import pandas as pd
import os
import rasterio
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


class GlaucomaDataset(Dataset):
    """
    Dataset for loading fundus images for the dataloader.
    
    Args:
        subset: pd.DataFrame - dataframe containing the images that the dataset
          should sample from
    """
    def __init__(self, subset: pd.DataFrame, stage: str):
        super().__init__()

        # get file paths and labels from subset and set stage (training, validation, test)
        self.subset = subset[['fundus', 'types']].reset_index().drop(columns='index')
        self.stage = stage
    
    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        # get file path from subset at index
        path = self.subset.iloc[index]['fundus']
        fixed_path = './data' + path

        # read image at path using rasterio
        with rasterio.open(fixed_path) as src:
            img = src.read()
        
        # get label from subset
        label = self.subset.iloc[index]['types']

        # apply Albumentations transforms for data augmentation
        if self.stage == "train":
            transforms = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.CLAHE(p=1.0),
                A.ToFloat(p=1.0),
                ToTensorV2(p=1.0),
            ])
        else:
            transforms = A.Compose([
                A.CLAHE(p=1.0),
                A.ToFloat(p=1.0),
                ToTensorV2(p=1.0)
            ])

        if img.shape == (3, 512, 512):
            img = np.transpose(img, (1, 2, 0))
        elif img.shape == (512, 3, 512):
            img = np.transpose(img, (0, 2, 1))
        img = transforms(image=img)["image"]

        if list(img.shape) == [512, 3, 512]:
            img = torch.permute(img, (1, 0, 2))
        elif list(img.shape) == [512, 512, 3]:
            img = torch.permute(img, (2, 0, 1))

        return img, label


class GlaucomaLDM(pl.LightningDataModule):
    """
    Pytorch Lightning Data Module for loading training, validation, and test datasets.

    Args:
        data: str - path to metadata.csv
        batch_size: int - number of images to include in each batch
        num_workers: int - number of workers to use
    """
    def __init__(self, data: str = "./data/metadata.csv", batch_size: int = 1, num_workers: int = 0):
        super().__init__()

        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

        # empty datasets
        self.train = None
        self.val = None
        self.test = None

    def prepare_data_per_node(self):
        pass

    def setup(self, stage):
        df = pd.read_csv(self.data, index_col=None)
        
        # drop rows without data
        df = df[(df['fundus'].notnull()) & (df['fundus'] != -1) & (df['types'] != -1)]
        df = df[['types', 'fundus']]
        df['fundus'] = df['fundus'].astype(str)
        df['types'] = df['types'].astype(int)

        # randomize data
        df = df.sample(frac=1, random_state=1)

        # split between healthy and glaucoma
        glaucoma_df = df[df['types'] == 1]
        healthy_df = df[df['types'] == 0]

        test_size = int(len(glaucoma_df) * 0.1)

        # get test sets for glaucoma and healthy and combine into one dataset
        healthy_test = healthy_df.tail(test_size)
        glaucoma_test = glaucoma_df.tail(test_size)

        glaucoma_trainval = glaucoma_df.head(len(glaucoma_df) - test_size)
        healthy_trainval = healthy_df.head(len(healthy_df) - test_size)

        combined_test = pd.concat([healthy_test, glaucoma_test])
        combined_test = combined_test.sample(frac=1, random_state=1)

        self.test = GlaucomaDataset(combined_test, stage="test")

        # get validation sets for glaucoma and healthy and combine
        val_size = int(len(glaucoma_trainval) * 0.1)

        glaucoma_val = glaucoma_trainval.tail(val_size)
        healthy_val = healthy_trainval.tail(val_size)

        combined_val = pd.concat([healthy_val, glaucoma_val])
        combined_val = combined_val.sample(frac=1, random_state=1)

        self.val = GlaucomaDataset(combined_val, stage="val")

        glaucoma_train = glaucoma_trainval.head(len(glaucoma_trainval) - val_size)
        healthy_train = healthy_trainval.head(len(glaucoma_train))

        combined_train = pd.concat([healthy_train, glaucoma_train])

        self.train = GlaucomaDataset(combined_train, stage="train")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False)