# Reproducibility

## Manuscript seeds

The manuscript uses three matched random seeds:

```text
666, 2026, 7777
```

`scripts/run_experiments.py` uses these seeds by default.

## CUB protocol

- Input: 448×448 pixels.
- Official CUB-200-2011 training/test split: 5994 / 5794 images.
- Source ResNet-50: 60 epochs; AdamW; backbone LR 2e-5; classifier LR 2e-4; weight decay 1e-4; 3-epoch warm-up followed by cosine decay.
- MVL-like: 50 epochs; AdamW; weight decay 1e-4; gradient clipping 0.5. Epochs 1–5 optimize the MVL-like module and `alpha_M`; later training additionally unfreezes layer3, layer4, and classifier.
- Standalone ENTO-like: 12 epochs with frozen backbone; AdamW LR 1e-4, weight decay 1e-4, cosine decay.
- Final ENTO–MVL: trained MVL-like source frozen; ENTO-like residual pathway optimized for 12 epochs with AdamW LR 1e-4, weight decay 1e-4, gradient clipping 0.5, batch size 2, and four-step gradient accumulation.
- Final ENTO–MVL checkpoint: fixed epoch 12; no early stopping and no official-test checkpoint selection.

## Data augmentation

Training uses 448-pixel bicubic resizing, ImageNet normalization, horizontal flipping (`p=0.5`), RandAugment (`N=2, M=7`), ColorJitter (brightness/contrast/saturation 0.10, hue 0.02), and random erasing (`p=0.10`) where enabled. Random erasing is disabled during the final ENTO adaptation. Evaluation uses no stochastic augmentation.

## RMS calibration

The fixed ENTO residual scale `s_E` is fitted using a class-stratified training-only subset. Official test images and their labels are not used for scale fitting or checkpoint selection.

## Training-only auxiliary terms

The released final joint trainer retains the auxiliary objectives used in the reported all-class experiment: ENTO auxiliary classification, training-only OOF confuser margin, source-margin preservation, source-error repair, soft-region overlap regularization, and residual L2 regularization. The OOF confuser map is generated entirely from training data.

## Environment

The manuscript environment was Python 3.9.24, PyTorch 2.5.1, CUDA 12.1, and cuDNN 9.1.
