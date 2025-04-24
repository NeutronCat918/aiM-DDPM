import torch
from model.diffusion_process import DiffusionModel
from trainer import Trainer
from model.utils import parser
from argparse import ArgumentParser, Namespace
import os
import json
from model.UNet import TwoResUNet,OneResUNet, AttentionUNet
from tester import Tester


#This file will need to be abstracted out of this code and read in later.
config_file={
  "model_name": "twores_128_1",
  "trainer_config": {
    "train_batch_size": 16, 
    "train_lr": 1e-4,
    "train_num_steps": 20000,
    "save_and_sample_every": 200,
    "num_samples": 4
  },
  "unet_config": {
    "model_mapping": "OneResUNet",
    "input": 64,
    "batch_size": 16,
    "dim_mults": [1, 2, 4, 8],
    "channels": 1
  },
  "diffusion_config": {
    "timesteps": 150,
    "betas_scheduler": "linear",
    "image_size": 128
  }
}

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_config = config_file.get("unet_config")
trainer_config = config_file.get("trainer_config")
diffusion_config = config_file.get("diffusion_config")
unet_config = config_file.get("unet_config")
trainer_config = config_file.get("trainer_config")
diffusion_config = config_file.get("diffusion_config")

unet_ = OneResUNet #Set Unet Type

model = unet_(
    dim=unet_config.get("input"),
    channels=unet_config.get("channels"),
    dim_mults=tuple(unet_config.get("dim_mults")),
).to(device)  #Make UNet Here

diffusion_model = DiffusionModel(
    model,
    image_size=diffusion_config.get("image_size"),
    beta_scheduler=diffusion_config.get("betas_scheduler"),
    timesteps=diffusion_config.get("timesteps"),
) #Make Diffusion model here

tester = Tester(diffusion_model=diffusion_model,    
   anom_folder='./Data/test',
    results_folder='./Results/Eval', 
    params_path=r"./Results/model-100.pt",
num_samples=100)



tester.eval()

