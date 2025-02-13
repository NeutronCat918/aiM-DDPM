
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

import torch




class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder,size):
        self.image_folder = image_folder
        self.images = os.listdir(self.image_folder)
        self.size=size

    # get sample
    def __getitem__(self, idx):
        img_file = f"{self.image_folder}/{self.images[idx]}"

        img_npy = np.load(img_file)[0:self.size,0:self.size]
        img_tensor= torch.Tensor(img_npy).reshape(1, self.size, self.size)
        return img_tensor,idx
        
    def __len__(self):
        return len(self.images)
