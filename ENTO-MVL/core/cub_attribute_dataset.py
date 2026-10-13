from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets

from data_pipeline import build_transforms, validate_imagefolder_root


CERTAINTY_WEIGHTS = {
    1: 0.0,   # not visible
    2: 0.33,  # guess
    3: 0.67,  # probably
    4: 1.0,   # definitely
}


def _read_id_path(path: Path) -> Dict[int, str]:
    result: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            image_id_text, relative_path = line.split(maxsplit=1)
            result[int(image_id_text)] = relative_path.replace("\\", "/")
    return result


def read_attribute_names(cub_root: str) -> Dict[int, str]:
    path = Path(cub_root) / "attributes" / "attributes.txt"
    if not path.exists():
        raise FileNotFoundError(
            "Cannot find CUB attribute-name file:\n"
            f"{path}\n"
            "Expected original CUB_200_2011/attributes/attributes.txt."
        )

    names: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            attribute_id_text, name = line.split(maxsplit=1)
            names[int(attribute_id_text)] = name
    return names


def select_attribute_ids(
    attribute_names: Dict[int, str],
    mode: str = "basic",
) -> List[int]:
    """
    Modes
    -----
    basic:
        Color, shape, size and length attributes. This is the recommended
        ENTO-like setting because local pattern/texture attributes are excluded.
    basic_plus_pattern:
        Basic attributes plus pattern attributes.
    all:
        All 312 CUB attributes.
    """
    mode = mode.lower()
    if mode not in {"basic", "basic_plus_pattern", "all"}:
        raise ValueError(
            "attribute_mode must be: basic, basic_plus_pattern, or all"
        )

    if mode == "all":
        return sorted(attribute_names)

    keywords = ["color", "shape", "size", "length"]
    if mode == "basic_plus_pattern":
        keywords.append("pattern")

    selected: List[int] = []
    for attribute_id, name in attribute_names.items():
        group = name.split("::", maxsplit=1)[0].lower()
        if any(keyword in group for keyword in keywords):
            selected.append(attribute_id)

    if not selected:
        raise RuntimeError(
            f"No attributes were selected with mode={mode}. "
            "Check attributes.txt."
        )
    return sorted(selected)


def load_image_attributes(
    cub_root: str,
    selected_attribute_ids: Sequence[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    labels_path = (
        Path(cub_root)
        / "attributes"
        / "image_attribute_labels.txt"
    )
    if not labels_path.exists():
        raise FileNotFoundError(
            "Cannot find CUB image-level attribute annotations:\n"
            f"{labels_path}"
        )

    selected_to_column = {
        attribute_id: column
        for column, attribute_id in enumerate(selected_attribute_ids)
    }
    num_attributes = len(selected_attribute_ids)

    targets: Dict[int, torch.Tensor] = {}
    weights: Dict[int, torch.Tensor] = {}

    with labels_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(
                    f"Malformed line {line_number} in {labels_path}: {line}"
                )

            image_id = int(parts[0])
            attribute_id = int(parts[1])
            is_present = float(parts[2])
            certainty_id = int(parts[3])

            column = selected_to_column.get(attribute_id)
            if column is None:
                continue

            if image_id not in targets:
                targets[image_id] = torch.zeros(
                    num_attributes,
                    dtype=torch.float32,
                )
                weights[image_id] = torch.zeros(
                    num_attributes,
                    dtype=torch.float32,
                )

            targets[image_id][column] = is_present
            weights[image_id][column] = CERTAINTY_WEIGHTS.get(
                certainty_id,
                0.0,
            )

    return targets, weights


class ImageIdResolver:
    def __init__(self, cub_root: str):
        cub_root_path = Path(cub_root)
        images_txt = cub_root_path / "images.txt"
        if not images_txt.exists():
            raise FileNotFoundError(
                f"Cannot find original CUB images.txt: {images_txt}"
            )

        self.id_to_path = _read_id_path(images_txt)
        self.path_to_id = {
            path.lower(): image_id
            for image_id, path in self.id_to_path.items()
        }

        self.basename_to_ids: Dict[str, List[int]] = {}
        self.stem_to_ids: Dict[str, List[int]] = {}
        for image_id, relative_path in self.id_to_path.items():
            path = Path(relative_path)
            basename = path.name.lower()
            stem = path.stem.lower()
            self.basename_to_ids.setdefault(basename, []).append(image_id)
            self.stem_to_ids.setdefault(stem, []).append(image_id)

    def resolve(
        self,
        sample_path: str,
        imagefolder_root: str,
    ) -> int:
        path = Path(sample_path)

        # BBox preparation script prefixes the original image ID:
        # 000001_Black_Footed_Albatross_0046_18.jpg
        match = re.match(r"^(\d{1,6})_", path.name)
        if match:
            image_id = int(match.group(1))
            if image_id in self.id_to_path:
                return image_id

        # Standard ImageFolder usually keeps class-folder/original-filename.
        tail = "/".join(path.parts[-2:]).replace("\\", "/").lower()
        if tail in self.path_to_id:
            return self.path_to_id[tail]

        basename = path.name.lower()
        basename_ids = self.basename_to_ids.get(basename, [])
        if len(basename_ids) == 1:
            return basename_ids[0]

        stem_ids = self.stem_to_ids.get(path.stem.lower(), [])
        if len(stem_ids) == 1:
            return stem_ids[0]

        raise KeyError(
            "Could not map ImageFolder sample to a CUB image ID:\n"
            f"sample={sample_path}\n"
            f"imagefolder_root={imagefolder_root}"
        )


class AttributeImageFolder(Dataset):
    """
    ImageFolder wrapper returning:
        image, class_label, attribute_target, attribute_weight, image_id
    """

    def __init__(
        self,
        imagefolder: datasets.ImageFolder,
        imagefolder_root: str,
        cub_root: str,
        selected_attribute_ids: Sequence[int],
        attribute_targets: Dict[int, torch.Tensor],
        attribute_weights: Dict[int, torch.Tensor],
    ) -> None:
        self.imagefolder = imagefolder
        self.root = imagefolder.root
        self.classes = imagefolder.classes
        self.class_to_idx = imagefolder.class_to_idx
        self.samples = imagefolder.samples
        self.targets = imagefolder.targets
        self.selected_attribute_ids = list(selected_attribute_ids)

        resolver = ImageIdResolver(cub_root)
        self.image_ids: List[int] = []
        errors: List[str] = []

        for sample_path, _ in self.samples:
            try:
                image_id = resolver.resolve(
                    sample_path,
                    imagefolder_root,
                )
                if image_id not in attribute_targets:
                    raise KeyError(
                        f"No selected attributes for image_id={image_id}"
                    )
                self.image_ids.append(image_id)
            except Exception as error:
                if len(errors) < 20:
                    errors.append(str(error))
                self.image_ids.append(-1)

        missing = sum(image_id < 0 for image_id in self.image_ids)
        if missing:
            examples = "\n---\n".join(errors)
            raise RuntimeError(
                f"Failed to map {missing}/{len(self.samples)} ImageFolder "
                "samples to CUB attributes.\n"
                f"Examples:\n{examples}"
            )

        self.attribute_targets = attribute_targets
        self.attribute_weights = attribute_weights

    def __len__(self) -> int:
        return len(self.imagefolder)

    def __getitem__(self, index: int):
        image, class_label = self.imagefolder[index]
        image_id = self.image_ids[index]
        return (
            image,
            class_label,
            self.attribute_targets[image_id].clone(),
            self.attribute_weights[image_id].clone(),
            image_id,
        )


def build_attribute_datasets(
    data_root: str,
    cub_root: str,
    img_size: int = 448,
    random_erasing: bool = True,
    attribute_mode: str = "basic",
):
    validate_imagefolder_root(data_root)
    train_transform, val_transform = build_transforms(
        img_size=img_size,
        random_erasing=random_erasing,
    )

    train_folder = datasets.ImageFolder(
        str(Path(data_root) / "train"),
        transform=train_transform,
    )
    val_folder = datasets.ImageFolder(
        str(Path(data_root) / "val"),
        transform=val_transform,
    )
    if train_folder.classes != val_folder.classes:
        raise RuntimeError("train/val class orders are different.")

    attribute_names_by_id = read_attribute_names(cub_root)
    selected_ids = select_attribute_ids(
        attribute_names_by_id,
        mode=attribute_mode,
    )
    targets, weights = load_image_attributes(
        cub_root,
        selected_ids,
    )

    train_set = AttributeImageFolder(
        train_folder,
        imagefolder_root=data_root,
        cub_root=cub_root,
        selected_attribute_ids=selected_ids,
        attribute_targets=targets,
        attribute_weights=weights,
    )
    val_set = AttributeImageFolder(
        val_folder,
        imagefolder_root=data_root,
        cub_root=cub_root,
        selected_attribute_ids=selected_ids,
        attribute_targets=targets,
        attribute_weights=weights,
    )

    selected_names = [
        attribute_names_by_id[attribute_id]
        for attribute_id in selected_ids
    ]
    return train_set, val_set, selected_names


def compute_attribute_pos_weight(
    dataset: AttributeImageFolder,
    max_pos_weight: float = 10.0,
) -> torch.Tensor:
    num_attributes = len(dataset.selected_attribute_ids)
    positive = torch.zeros(num_attributes, dtype=torch.float64)
    negative = torch.zeros(num_attributes, dtype=torch.float64)

    for image_id in dataset.image_ids:
        target = dataset.attribute_targets[image_id].double()
        weight = dataset.attribute_weights[image_id].double()
        positive += weight * target
        negative += weight * (1.0 - target)

    pos_weight = negative / positive.clamp_min(1.0)
    pos_weight = pos_weight.clamp(0.5, float(max_pos_weight))
    return pos_weight.float()
