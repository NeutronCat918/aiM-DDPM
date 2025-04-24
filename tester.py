import torch
from tqdm import tqdm
from torchvision import utils
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from pathlib import Path
from model import utils, diffusion_process
from Datasets import MyDataset
import os
import glob
import numpy as np

class Tester:
    def __init__(
        self,
        diffusion_model: diffusion_process.DiffusionModel,
        anom_folder: str,
        params_path: str,
        results_folder: str,
        *,
        num_samples: int = 4,
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
    ) -> None:
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.step = 0
        self.anom_folder = anom_folder

        self.num_samples = num_samples

        self.image_size = diffusion_model.image_size

        self.ds = MyDataset(
            anom_folder, self.image_size 
        )
        self.params = params_path
        self.dl = utils.cycle(
            DataLoader(
                self.ds, batch_size=self.num_samples, shuffle=True, pin_memory=True
            )
        )
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to("cuda" if torch.cuda.is_available() else "cpu")

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

    @property
    def device(self) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ['TORCH_USE_CUDA_DSA'] = "1"
        return device

    def load(self) -> None:
        data = torch.load(self.params, map_location=self.device, weights_only=True)
        self.model.model.load_state_dict(data["model"], strict=False)
        self.step = data["step"]
        self.ema.load_state_dict(data["ema"])
        print("parameters loaded")

        if "version" in data:
            print(f"loading from version {data['version']}")
            self.version = data['version']

    def eval(self) -> None:
        self.load()

        loaded_data = next(self.dl)

        anom_image = loaded_data[0].to(self.device)
        anom_indexies = loaded_data[1].detach().cpu().numpy()
        print("data loaded")
        self.ema.ema_model.eval()
        filelist = os.listdir(glob.glob(self.anom_folder)[0])

        with torch.inference_mode():       
            denoised_imgs = self.ema.ema_model.sample(
                initial_image = anom_image
            )
            sampled_imgs = self.ema.ema_model.sample(
                batch_size = self.num_samples
            )

            for ix, sampled_img in enumerate(sampled_imgs):
                np.save(str(self.results_folder)+f"/Sample/{ix}.npy",sampled_img.detach().cpu().numpy().reshape((self.image_size,self.image_size)))
    
            for ix, denoised_img in enumerate(denoised_imgs):
                np.save(str(self.results_folder)+f"/Denoised/{filelist[anom_indexies[ix]]}",denoised_img.detach().cpu().numpy().reshape((self.image_size,self.image_size)))
    
            for ix,data in enumerate(anom_image):
                np.save(str(self.results_folder)+f"/Anomaly/{filelist[anom_indexies[ix]]}",data.detach().cpu().numpy().reshape((self.image_size,self.image_size)))

# Initialize the Tester class and call the eval method
if __name__ == "__main__":
    diffusion_model = diffusion_process.DiffusionModel()  # Replace with your model initialization
    tester = Tester(
        diffusion_model=diffusion_model,
        anom_folder="path/to/anom_folder",
        params_path="path/to/params.pth",
        results_folder="path/to/results_folder"
    )
    tester.eval()

