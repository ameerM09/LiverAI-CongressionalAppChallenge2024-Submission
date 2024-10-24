from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss

import torch
from data_preprocessing import preprocess_data
from settings import train

main_dir = "G:/DL for Liver Segmentation/final_nifti_files"
data_input_path = preprocess_data(main_dir = main_dir, cache=True)

device = torch.device("cuda:0")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model_dir = "G:/DL for Liver Segmentation/results" 

loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_input_path, loss_function, optimizer, 100, model_dir)