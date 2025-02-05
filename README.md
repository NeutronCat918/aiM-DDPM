```
.
└── DDPM/
    ├── Data/
    │   ├── Anomaly/
    │   │   ├── 1.npy
    │   │   ├── 2.npy
    │   │   └── ...
    │   └── Ground Truth/
    │       ├── 1.npy
    │       ├── 2.npy
    │       └── ...
    ├── Results/
    │   ├── Eval/
    │   │   ├── Anomaly
    │   │   ├── Denoised
    │   │   ├── Ground Truth
    │   │   └── Sample
    │   └── Train/
    │       ├── 1/
    │       │   ├── Denoised
    │       │   ├── GT
    │       │   └── Sampled
    │       ├── 2/
    │       │   ├── Denoised
    │       │   ├── GT
    │       │   └── Sampled
    │       ├── ...
    │       ├── model-1.pt
    │       ├── model-2.pt
    │       └── ...
    ├── model/
    │   ├── UNet.py
    │   ├── components.py
    │   ├── diffusion_process.py
    │   ├── scheduler.py
    │   └── utils.py
    ├── Datasets.py
    ├── tester.py
    └── trainer.py
```
