from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_imagefolder_root(data_root: str) -> None:
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "data_root must contain train/ and val/ folders.\n"
            f"Missing under: {root}"
        )


def build_transforms(
    img_size: int = 448,
    random_erasing: bool = True,
):
    """
    The same preprocessing must be used for every baseline/plugin/KD experiment.

    Validation deliberately uses direct square resize because the user's earlier
    checkpoints were evaluated this way. Do not mix this with center-crop results.
    """
    train_ops = [
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if random_erasing:
        train_ops.append(
            transforms.RandomErasing(
                p=0.10,
                scale=(0.02, 0.12),
                ratio=(0.5, 2.0),
                value="random",
            )
        )

    val_ops = [
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def build_datasets(
    data_root: str,
    img_size: int = 448,
    random_erasing: bool = True,
) -> Tuple[Dataset, Dataset]:
    validate_imagefolder_root(data_root)
    train_tf, val_tf = build_transforms(
        img_size=img_size,
        random_erasing=random_erasing,
    )
    train_set = datasets.ImageFolder(
        str(Path(data_root) / "train"),
        transform=train_tf,
    )
    val_set = datasets.ImageFolder(
        str(Path(data_root) / "val"),
        transform=val_tf,
    )
    if train_set.classes != val_set.classes:
        raise RuntimeError("train/val class orders are different.")
    return train_set, val_set
