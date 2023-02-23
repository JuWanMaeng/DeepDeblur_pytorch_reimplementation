# readme

cite: https://github.com/SeungjunNah/DeepDeblur_release

# 1. Dataset

## GoProLarge

train.json  → 1893

val.json     → 210

# 2. Model

| Model | MsResNet |
| --- | --- |
| rgb_range | 255 |
| n_resblock | 19 |
| n_feats | 64 |
| n_kernel_size | 5 |
| n_sclaes | 3 |
| Gaussian_pyramid | True |

# 3. Optimizer

| optimizer | Adam(1e-4) |
| --- | --- |
| scheduler | multstep (500,750,900), gamma: 0.5 |

# 4. Loss

![Untitled](https://github.com/JuWanMaeng/DeepDeblur_pytorch_reimplementation/blob/main/figure/Untitled%201.png)

![Untitled](https://github.com/JuWanMaeng/DeepDeblur_pytorch_reimplementation/blob/main/figure/Untitled%202.png)

![Untitled](https://github.com/JuWanMaeng/DeepDeblur_pytorch_reimplementation/blob/main/figure/Untitled.png)

| content loss | L2 |
| --- | --- |
| adversial loss | lamda: 10e-4 |

# 5. experiment (k=3)

|  | val | test | paper |
| --- | --- | --- | --- |
| psnr | 30.2516 | 28.08 (max:28.59) | 29.08 |
| ssim | 0.8921 |  | 0.9135 |
