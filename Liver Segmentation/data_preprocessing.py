from monai.transforms import (
    Compose, 
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)

import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset, CacheDataset
from glob import glob
import os

def preprocess_data(main_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 64], cache=True):
    training_images = sorted(glob(os.path.join(main_dir, "images", "*.nii.gz")))
    training_labels = sorted(glob(os.path.join(main_dir, "labels", "*.nii.gz")))

    val_images = sorted(glob(os.path.join(main_dir, "imagesVal", "*.nii.gz")))
    val_labels = sorted(glob(os.path.join(main_dir, "labelsVal", "*.nii.gz")))

    training_files = [{"vol": image_num, "seg": label_num} for image_num, label_num in zip(training_images, training_labels)]
    val_files = [{"vol": image_num, "seg": label_num} for image_num, label_num in zip(val_images, val_labels)]

    training_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),
            
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"], reader="NibabelReader"),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )

    if cache:
        train_ds = CacheDataset(data = training_files, transform=training_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data = val_files, transform=val_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=training_files, transform=training_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=val_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader