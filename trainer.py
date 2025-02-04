import os
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

class Trainer:
    def __init__(
        self,
        diffusion_model: diffusion_process.DiffusionModel,
        folder: str,
        results_folder: str,
        *,
        train_batch_size: int = 16,
        augment_horizontal_flip: bool = True,
        train_lr: float = 1e-4,
        train_num_steps: int = 100,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        num_samples: int = 4,
        save_best_and_latest_only: bool = False,
        best_params=None,
    ) -> None:
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.step = 0

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.ds = MyDataset(
            folder, self.image_size 
        )
        self.dl = utils.cycle(
            DataLoader(
                self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True
            )
        )
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.save_best_and_latest_only = save_best_and_latest_only
        self.best_params=best_params

    @property
    def device(self) -> str:
        device="cuda" if torch.cuda.is_available() else "cpu"
        return device

    def save(self, milestone: int) -> None:
        data = {
            "step": self.step,
            "model": self.model.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "version": "1.0",
        }

        torch.save(data, str(self.results_folder)+f"/model-{milestone}.pt")
        print("saved")

    def load(self) -> None:
        if self.best_params is not None and os.path.exists(self.best_params):
            data = torch.load(
                self.best_params,
                map_location=self.device,
            )
            self.model.model.load_state_dict(data["model"])

            self.step = data["step"]
            self.opt.load_state_dict(data["opt"])
            self.ema.load_state_dict(data["ema"])

            if "version" in data:
                print(f"loading from version {data['version']}")
        else:
            print("No saved model found. Starting training from scratch.")

    def train(self, ifcontinue: bool = False) -> None:
        if ifcontinue:
            self.load()
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                data = next(self.dl)[0].to(self.device)
                index=next(self.dl)[1].detach().cpu().numpy()
                loss = self.model(data)
                total_loss += loss.item()

                loss.backward()

                pbar.set_description(f"loss: {total_loss:.4f}")

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()

                    with torch.inference_mode():
                        milestone = self.step // self.save_and_sample_every
                        denoised_imgs = self.ema.ema_model.sample(
                            initial_image=data
                        )
                        sampled_imgs = self.ema.ema_model.sample(
                            batch_size=self.num_samples
                        )

                    for ix, sampled_img in enumerate(sampled_imgs):
                        torchvision.utils.save_image(
                            sampled_img,
                            str(self.results_folder)+f"/sample-{milestone}-{ix}.png",
                        )
                    for ix, denoised_img in enumerate(denoised_imgs):
                        torchvision.utils.save_image(
                            denoised_img,
                            str(self.results_folder)+f"/denoise-{milestone}-{index[ix]}.png",
                        )
                        np.save(str(self.results_folder)+f"/denoise/{milestone}-{index[ix]}.npy",denoised_img.detach().cpu().numpy())
                        np.save(str(self.results_folder)+f"/denoise/gt_{milestone}-{index[ix]}.npy",data[ix,0,:,:].detach().cpu().numpy())

                    self.save(milestone)
                pbar.update(1)
