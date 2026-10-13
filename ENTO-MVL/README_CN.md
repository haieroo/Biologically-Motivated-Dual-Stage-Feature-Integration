# ENTO–MVL 公开代码

本仓库对应修订稿中的最终 ENTO–MVL 模型。最终结构在**完整类别空间**上进行联合决策，不包含硬类别候选掩码或重排序候选集。

## 最终模型

- **ENTO-like**：ResNet-50 `layer2` (`f2`, 512 channels) → 4 个可学习软空间区域 → 区域外观表征 → 独立分类头与有界残差头 `Delta_E`。
- **MVL-like**：ResNet-50 `layer3` → 局部细节/统计编码 → 残差适配 → 类别证据；训练完成的 MVL-like 模型提供完整类别得分 `M`。
- **ENTO–MVL 联合模型**：对所有类别执行

```text
M   = z_base + alpha_M * z_M
z_A = M + alpha * s_E * Delta_E
```

论文设置中 `alpha = 0.20`。`s_E` 为仅利用训练图像拟合的固定 RMS 校准因子。最终 12 个 epoch 的联合适配阶段冻结已训练的 MVL-like 主路径，只训练 ENTO-like 残差路径。

评估脚本中的 `Top-5` 仅表示标准 Top-5 accuracy 指标，不参与模型结构中的类别筛选。

## 论文三随机种子复现

默认使用论文中的三个随机种子：`666, 2026, 7777`。

```bash
python scripts/run_experiments.py \
  --data-root /path/to/CUB_ImageFolder \
  --cub-root /path/to/CUB_200_2011 \
  --clean-checkpoint /path/to/resnet50_source.pth \
  --output-root outputs
```

最终 checkpoint：

```text
outputs/results/seed<SEED>/ento_mvl/final_ento_mvl.pth
```

最终联合训练脚本保留公开实验实际使用的辅助训练项，包括最终分类损失、ENTO 辅助分类、训练集 OOF confuser margin、源模型 margin 保持、错误修复、软区域重叠正则以及残差 L2 正则。这些训练项均不会限制最终分类时的类别集合。

主要代码见 `core/models_ento_mvl.py`、`training/train_ento_mvl.py` 和 `scripts/run_experiments.py`。
