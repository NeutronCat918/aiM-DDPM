import torch
from model.diffusion_process import DiffusionModel
from trainer import Trainer
from model.utils import parser
from argparse import ArgumentParser, Namespace
import os
import json
from model.UNet import TwoResUNet,OneResUNet, AttentionUNet
from tester import Tester


config_file={
  "model_name": "twores_128_1",
  "trainer_config": {
    "train_batch_size": 16, 
    "train_lr": 1e-5,
    "train_num_steps": 10000,
    "save_and_sample_every": 100,
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
    "timesteps": 100,
    "betas_scheduler": "linear",
    "image_size": 128
  }
}

  }

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

#make input and output paths relative to make it easier to to add code to your desired directory
trainer = Trainer(
    diffusion_model=diffusion_model,
    folder='./Data/Ground-Truth',
    results_folder='./Results',
    train_batch_size=trainer_config.get("train_batch_size"),
    train_lr=trainer_config.get("train_lr"),
    train_num_steps=trainer_config.get("train_num_steps"),
    save_and_sample_every=trainer_config.get("save_and_sample_every"),
    num_samples=trainer_config.get("num_samples"),
    best_params='./Results/best-model-yet.pt')

tester = Tester(diffusion_model=diffusion_model,    
   anom_folder='./Data/Anomaly',
    results_folder='./Results/Eval', 
    params_path=r"./Results/best-model-yet.pt",
num_samples=16)



tester.eval()
