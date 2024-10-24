import os
from glob import glob
from pathlib import Path

from monai.transforms import(
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm

from monai.inferers import sliding_window_inference

from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric

import torch

import matplotlib.pyplot as plt

def get_test_file_by_name(file_name, test_files):
    for file in test_files:
        if os.path.basename(file["vol"]) == file_name:
            return file
    return None

def perform_segmentation(primary_dir, model_dir, test_file_name):
    path_test_volumes = sorted(glob(os.path.join(primary_dir, "imagesVal", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(primary_dir, "labelsVal", "*.nii.gz")))

    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]
    specific_test_file = get_test_file_by_name(test_file_name, test_files)
    
    if specific_test_file:
        test_files = [specific_test_file]
    else:
        raise ValueError(f"File {test_file_name} not found in test files.")
    
    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstD(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),   
            ToTensord(keys=["vol", "seg"]),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, "liver_segmentation.pth"), map_location = device))

    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    sw_batch_size = 4
    roi_size = (128, 128, 64)

    with torch.no_grad():
        for test_data in test_loader:
            test_images = test_data["vol"].to(device)
            test_labels = test_data["seg"].to(device)
            
            test_outputs = sliding_window_inference(test_images.to(device), roi_size, sw_batch_size, model)
            sigmoid_activation = Activations(sigmoid=True)
            test_outputs = sigmoid_activation(test_outputs)
            test_outputs = test_outputs > 0.53
            
            dice_metric(y_pred=test_outputs, y=test_labels)

            plot_segmentation_slices(test_data, test_outputs)
        
        metric = dice_metric.aggregate().item() * 100
        dice_metric.reset()

    return metric

def plot_segmentation_slices(test_patient, test_outputs):
    for i in range(0, 50):
        plt.figure("check", (18, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Image #{i}")
        plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title(f"Prediction #{i}")
        plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])

        plt.savefig(str(os.path.join(Path.home(), "Downloads", f"output_{i}.png")))
        plt.show()
        plt.close()