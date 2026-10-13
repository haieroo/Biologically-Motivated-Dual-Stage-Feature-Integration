# Architecture

## Backbone feature sources

The model uses a pretrained ResNet-50 backbone as the visual substrate. `layer2` and `layer3` are operational feature sources rather than biological homologues of ENTO and MVL.

```text
f2 = ResNet50.layer2(x)    # 512-channel feature map → ENTO-like
f3 = ResNet50.layer3(x)    # deeper feature map       → MVL-like
f4 → GAP → classifier      # baseline class scores
```

## ENTO-like pathway

`f2` is processed by four learned soft spatial regions. Their responses are adaptively aggregated and projected to an ENTO-like representation. A standard classification head supports standalone evaluation, while a separate bounded residual head produces `Delta_E` for the final joint model.

## MVL-like pathway

The MVL-like pathway uses layer3 features, local 3×3 operations, and first-/second-order feature statistics. Its class evidence is integrated with the baseline classifier scores:

```text
M = z_base + alpha_M * z_M
```

## Joint integration

The final model adds calibrated ENTO-like residual evidence to `M` for every class:

```text
z_A,k = M_k + alpha * s_E * Delta_E,k,   k = 1,...,N
```

No class-selection mask is used. There is no ENTO×MVL interaction head and no additional MVL residual at the final integration step.

Implementation: `core/models_ento_mvl.py`.
