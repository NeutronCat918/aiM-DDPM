import torch
from tqdm import tqdm
from torchvision import utils
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
from model import utils,diffusion_process
# from model.diffusion_process import DiffusionModel
from Datasets import MyDataset
import torchvision

class Tester:
    def __init__(
        self,
        diffusion_model: diffusion_process.DiffusionModel,
        anom_folder: str,
        params_path:str,
        results_folder: str,
        *,
        train_batch_size: int = 16,
        save_and_sample_every: int = 1000,
        num_samples: int = 4,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        save_best_and_latest_only: bool = False,
    ) -> None:
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.step = 0

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every


        self.image_size = diffusion_model.image_size

        self.ds = MyDataset(
            anom_folder, self.image_size 
        )
        self.params=params_path
        self.dl = utils.cycle(
            DataLoader(
                self.ds, batch_size=self.num_samples, shuffle=True, pin_memory=True
            )
        )
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to("cuda" if torch.cuda.is_available() else "cpu")

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.save_best_and_latest_only = save_best_and_latest_only

    

    @property
    def device(self) -> str:
        device="cuda" if torch.cuda.is_available() else "cpu"
        return device


    def load(self) -> None:
        data = torch.load(self.params,
            map_location=self.device,
        )


        # del data["ema"]["online_model.log_one_minus_alphas_cumprod"]
        # del data["ema"]["online_model.sqrt_recip_alphas_cumprod"]
        # del data["ema"]["online_model.sqrt_recipm1_alphas_cumprod"]
        # del data["ema"]["ema_model.log_one_minus_alphas_cumprod"]
        # del data["ema"]["ema_model.sqrt_recip_alphas_cumprod"]
        # del data["ema"]["ema_model.sqrt_recipm1_alphas_cumprod"]

        self.model.model.load_state_dict(data["model"])

        self.step = data["step"]
        self.ema.load_state_dict(data["ema"])
        print("parameter loaded")

        if "version" in data:
            print(f"loading from version {data['version']}")
            self.version=data['version']

    def eval(self) -> None:
        self.load()


        anom_image = next(self.dl)[0].to(self.device)
        anom_index=next(self.dl)[1].detach().cpu().numpy()
        print("data loaded")
        self.ema.ema_model.eval()

        with torch.inference_mode():       
          denoised_imgs = self.ema.ema_model.sample(
                            # batch_size=self.num_samples
                            initial_image=anom_image
                        )
          sampled_imgs = self.ema.ema_model.sample(
                            batch_size=self.num_samples
                        )

          for ix, sampled_img in enumerate(sampled_imgs):
            torchvision.utils.save_image(
              sampled_img,
              str(self.results_folder)+f"/sample-test-{ix}.png",
                        )
          for ix, denoised_img in enumerate(denoised_imgs):
            torchvision.utils.save_image(
              denoised_img,
              str(self.results_folder)+f"/denoise-test-{anom_index[ix]}.png",
                        )
            np.save(str(self.results_folder)+f"/denoise/denoised-{anom_indexies[ix]}.npy",denoised_img.detach().cpu().numpy())
            np.save(str(self.results_folder)+f"/denoise/anom-{anom_indexies[ix]}.npy",anom_img[ix,0,:,:].detach().cpu().numpy())

        
