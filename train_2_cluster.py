#Objective of this python script: Turn the code written in Runner.ipynb into a submittable script for the DCC cluster.

#import modules
import torch #this module needs to be part of your conda enviornment
from model.diffusion_process import DiffusionModel #this module is defined within the directory
from trainer import Trainer #this module is defined within the directory
from model.utils import parser #this module is defined within the directory
from argparse import ArgumentParser, Namespace #this module needs to be part of your conda enviornment
import os #standard python library module
import json #standard python library module
from model.UNet import TwoResUNet,OneResUNet #this module is defined within the directory
#from tester import Tester #not running testing on this code, training only.


#This file will need to be abstracted out of this code and read in later.
config_file={
  "model_name": "twores_128_1",
  "trainer_config": {
    "train_batch_size": 16, #currently 1680 images in Ground-Truth, which means there are 105 batches per epoch
    "train_lr": 1e-4,
    "train_num_steps": 1000,
    "save_and_sample_every": 300,
    "num_samples": 4
  },
  "unet_config": {
    "model_mapping": "TwoResUNet",
    "input": 64,
    "batch_size": 16,
    "dim_mults": [1, 2, 4, 8],
    "channels": 1
  },
  "diffusion_config": {
    "timesteps": 500, #downsized for time efficiency
    "betas_scheduler": "linear",
    "image_size": 128
  }
}

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


unet_config = config_file.get("unet_config")
trainer_config = config_file.get("trainer_config")
diffusion_config = config_file.get("diffusion_config")

unet_ = TwoResUNet #Set Unet Type

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
    best_params='./Results/model-X.pt')


trainer.train(ifcontinue=True)