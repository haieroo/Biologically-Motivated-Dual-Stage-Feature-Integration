# ENTO–MVL

Official implementation of the ENTO–MVL model described in “Biologically Motivated Dual-Stage Feature Integration for Fine-Grained Visual Recognition.”  
The final architecture uses all-class joint integration. No hard candidate-selection or class-restriction operation is used in the final prediction.

## Final architecture

The implementation follows the final architecture described in the manuscript and uses **all-class joint integration**. There is no hard class-candidate mask or re-ranking set in the final architecture.

- **ENTO-like:** ResNet-50 `layer2` feature map (`f2`, 512 channels) → four learned soft spatial regions → regional appearance representation → standalone ENTO-like classifier and bounded residual head `Delta_E`.
- **MVL-like:** ResNet-50 `layer3` features → local-detail/statistical encoder → residual adapter → class evidence. The trained MVL-like source produces the complete class-score output `M`.
- **Joint ENTO–MVL:** the calibrated ENTO-like residual is added to `M` for every class:

```text
M   = z_base + alpha_M * z_M
z_A = M + alpha * s_E * Delta_E      # all N classes
```

The paper setting uses `alpha = 0.20`. `s_E` is a fixed RMS-based calibration factor fitted from training images only. The MVL-like source is frozen during the final 12-epoch ENTO residual adaptation.

`Top-5` in evaluation scripts refers only to the standard Top-5 accuracy metric; it is not used as an architectural restriction.

## Paper training schedule

1. Train the ResNet-50 source model.
2. Train the MVL-like model for 50 epochs.
3. Build a training-only out-of-fold confuser map used by the auxiliary margin objective.
4. Train the standalone ENTO-like model for 12 epochs with the backbone frozen.
5. Freeze the trained MVL-like source and train the final ENTO–MVL residual pathway for 12 epochs.
6. Fit `s_E` using training images only, then evaluate once on the official test split.

The final joint-training code retains the auxiliary objectives used in the reported all-class experiments: final classification loss, ENTO auxiliary classification, an OOF confuser margin, source-margin preservation, source-error repair, soft-region overlap regularization, and residual L2 regularization. None of these terms removes classes from the final decision space.

## Reproduce the CUB experiment

```bash
python scripts/run_experiments.py \
  --data-root /path/to/CUB_ImageFolder \
  --cub-root /path/to/CUB_200_2011 \
  --clean-checkpoint /path/to/resnet50_source.pth \
  --output-root outputs
```

Final checkpoints are written to:

```text
outputs/results/seed<SEED>/ento_mvl/final_ento_mvl.pth
```

## Main files

```text
core/models_ento_mvl.py                 final all-class ENTO–MVL model
core/models_unified_mvllike.py          shared ENTO-like/MVL-like components
training/train_resnet50_baseline.py     source ResNet-50 recipe
training/train_mvl_like.py              MVL-like training
training/train_ento_like.py             standalone ENTO-like training
training/train_ento_mvl.py              final joint training
evaluation/eval_four_model_ablation.py  ResNet-50 / ENTO-like / MVL-like / ENTO–MVL
evaluation/eval_behavior_neural_rsa.py  behavior and neural RSA
evaluation/eval_robustness.py           perturbation evaluation
```

See `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md`, and `docs/INPUT_FORMATS.md` for additional details.

## Environment

The manuscript experiments used Python 3.9.24, PyTorch 2.5.1, CUDA 12.1, and cuDNN 9.1. The accompanying `environment.yml` and `requirements.txt` provide the package specification.

## Data

CUB-200-2011 and the external fine-grained benchmarks should be obtained from their original sources. Example neural-manifest files are provided under `examples/`. Neural and behavioral RDMs are not bundled unless redistribution is permitted by the data-availability statement accompanying the manuscript.

## License

See `LICENSE`.
