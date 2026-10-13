from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data_pipeline import IMAGENET_MEAN, IMAGENET_STD


def install_selector_preserving_transform(train_set, img_size: int) -> None:
    """Mild augmentation used by the leakage-free Compact-MVL OOF routine."""
    train_set.imagefolder.transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=4.0,
                translate=(0.025, 0.025),
                scale=(0.97, 1.03),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
