from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _trim_trailing_empty_cells(row: Sequence[str]) -> List[str]:
    cells = [str(value).strip() for value in row]
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _load_numeric_text_matrix(source: Path) -> np.ndarray:
    """
    Load either:
      1. a plain numeric N x N CSV/TXT matrix; or
      2. a labeled (N+1) x (N+1) table whose top-left cell is ImageID,
         first row contains column labels, and first column contains row labels.

    Also supports a header-only row or a label-only first column.
    """
    with source.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        sample = handle.read(4096)
        handle.seek(0)

        # Most RDM files are comma-separated. Fall back to whitespace if no
        # comma is present in the beginning of the file.
        if "," in sample:
            reader = csv.reader(handle)
            rows = [
                _trim_trailing_empty_cells(row)
                for row in reader
            ]
        else:
            rows = [
                line.strip().split()
                for line in handle
            ]

    rows = [row for row in rows if any(cell != "" for cell in row)]
    if not rows:
        raise ValueError(f"RDM text file is empty: {source}")

    candidates = []
    errors = []

    # Try plain numeric, header row only, first-column labels only, and both
    # header row + first-column labels.
    for row_offset, column_offset in (
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ):
        selected_rows = rows[row_offset:]
        if not selected_rows:
            continue

        selected = [
            row[column_offset:]
            for row in selected_rows
        ]
        if not selected or any(len(row) == 0 for row in selected):
            continue

        widths = {len(row) for row in selected}
        if len(widths) != 1:
            errors.append(
                f"offset=({row_offset},{column_offset}) ragged widths={sorted(widths)}"
            )
            continue

        try:
            array = np.asarray(
                [
                    [float(cell) for cell in row]
                    for row in selected
                ],
                dtype=np.float32,
            )
        except ValueError as exc:
            errors.append(
                f"offset=({row_offset},{column_offset}) nonnumeric: {exc}"
            )
            continue

        if array.ndim == 2 and array.shape[0] == array.shape[1]:
            candidates.append(
                (
                    int(array.shape[0]),
                    row_offset,
                    column_offset,
                    array,
                )
            )

    if not candidates:
        preview = rows[0][:6]
        raise ValueError(
            "Could not find a square numeric block in RDM file. "
            "Supported formats are a plain numeric square matrix or a table "
            "with one header row and/or one row-label column. "
            f"First row preview: {preview}. Attempts: {errors[:4]}"
        )

    # A plain matrix also creates smaller square candidates after dropping its
    # first row/column. Select the largest valid square block.
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, _, _, matrix = candidates[0]
    return matrix


def load_square_matrix(path: str, mat_key: Optional[str] = None) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(source)
    elif suffix in {".csv", ".txt"}:
        matrix = _load_numeric_text_matrix(source)
    elif suffix == ".mat":
        from scipy.io import loadmat

        content = loadmat(source)
        if mat_key is not None:
            if mat_key not in content:
                raise KeyError(
                    f"MAT key '{mat_key}' not found. Keys: {list(content)}"
                )
            matrix = content[mat_key]
        else:
            candidates = []
            for key, value in content.items():
                if key.startswith("__"):
                    continue
                array = np.asarray(value)
                if array.ndim == 2 and array.shape[0] == array.shape[1]:
                    candidates.append((key, array))
            if len(candidates) != 1:
                raise ValueError(
                    "MAT file must contain exactly one square numeric matrix "
                    "or provide --ento_rdm_key/--mvl_rdm_key. "
                    f"Candidates: {[key for key, _ in candidates]}"
                )
            matrix = candidates[0][1]
    else:
        raise ValueError(
            f"Unsupported RDM extension: {suffix}. Use CSV, TXT, NPY or MAT."
        )

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"RDM must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("RDM contains NaN or Inf.")

    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def load_neural_manifest(
    path: str,
    image_root: Optional[str] = None,
) -> List[Dict[str, str]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    root = Path(image_root) if image_root else source.parent
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Neural manifest has no header.")
        required = {"stimulus_id", "image_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Neural manifest missing columns {sorted(missing)}. "
                "Required: stimulus_id,image_path"
            )
        for row in reader:
            raw_path = Path(row["image_path"])
            resolved = raw_path if raw_path.is_absolute() else root / raw_path
            if not resolved.exists():
                raise FileNotFoundError(
                    f"Neural image not found: {resolved}"
                )
            rows.append(
                {
                    "stimulus_id": str(row["stimulus_id"]),
                    "image_path": str(resolved),
                }
            )

    if len(rows) < 4:
        raise ValueError("At least four neural stimuli are required.")
    stimulus_ids = [row["stimulus_id"] for row in rows]
    if len(set(stimulus_ids)) != len(stimulus_ids):
        raise ValueError("stimulus_id values must be unique.")
    return rows


def validate_neural_inputs(
    manifest_rows: Sequence[Dict[str, str]],
    ento_rdm: np.ndarray,
    mvl_rdm: np.ndarray,
) -> None:
    n = len(manifest_rows)
    if ento_rdm.shape != (n, n):
        raise ValueError(
            f"ENTO RDM shape {ento_rdm.shape} does not match manifest N={n}."
        )
    if mvl_rdm.shape != (n, n):
        raise ValueError(
            f"MVL RDM shape {mvl_rdm.shape} does not match manifest N={n}."
        )


def upper_triangle_vector(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.size(0)
    indices = torch.triu_indices(
        n,
        n,
        offset=1,
        device=matrix.device,
    )
    return matrix[indices[0], indices[1]]


def cosine_rdm(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(
        embeddings.float(),
        dim=1,
        eps=1e-8,
    )
    return 1.0 - normalized @ normalized.t()


def pearson_correlation(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    first = first.float()
    second = second.float()
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.sqrt(
        first.square().sum() * second.square().sum()
    ).clamp_min(1e-8)
    return (first * second).sum() / denominator


def rdm_correlation(
    embeddings: torch.Tensor,
    target_rdm: torch.Tensor,
) -> torch.Tensor:
    network_vector = upper_triangle_vector(
        cosine_rdm(embeddings)
    )
    target_vector = upper_triangle_vector(target_rdm)
    return pearson_correlation(network_vector, target_vector)


def rdm_alignment_loss(
    embeddings: torch.Tensor,
    target_rdm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    correlation = rdm_correlation(embeddings, target_rdm)
    return 1.0 - correlation, correlation


def transform_targets_for_mode(
    ento_rdm: torch.Tensor,
    mvl_rdm: torch.Tensor,
    mode: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "true_neural" or mode == "no_neural":
        return ento_rdm, mvl_rdm
    if mode == "swapped_neural":
        return mvl_rdm, ento_rdm
    if mode == "shuffled_neural":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 9173)
        permutation_ento = torch.randperm(
            ento_rdm.size(0),
            generator=generator,
        )
        permutation_mvl = torch.randperm(
            mvl_rdm.size(0),
            generator=generator,
        )
        shuffled_ento = ento_rdm[
            permutation_ento
        ][:, permutation_ento]
        shuffled_mvl = mvl_rdm[
            permutation_mvl
        ][:, permutation_mvl]
        return shuffled_ento, shuffled_mvl
    raise ValueError(mode)


def load_confuser_map(
    path: Optional[str],
    num_classes: int,
) -> Optional[torch.Tensor]:
    if not path:
        return None
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    content = json.loads(source.read_text(encoding="utf-8"))

    rows = []
    max_neighbors = 0
    for class_index in range(num_classes):
        values = content.get(str(class_index), [])
        values = [
            int(value)
            for value in values
            if int(value) != class_index
            and 0 <= int(value) < num_classes
        ]
        rows.append(values)
        max_neighbors = max(max_neighbors, len(values))

    if max_neighbors == 0:
        raise ValueError("Confuser map contains no valid neighbors.")

    tensor = torch.full(
        (num_classes, max_neighbors),
        -1,
        dtype=torch.long,
    )
    for class_index, values in enumerate(rows):
        if values:
            tensor[class_index, : len(values)] = torch.tensor(values)
    return tensor


def choose_confuser(
    labels: torch.Tensor,
    source_logits: torch.Tensor,
    confuser_map: Optional[torch.Tensor],
) -> torch.Tensor:
    if confuser_map is None:
        masked = source_logits.float().clone()
        masked.scatter_(1, labels[:, None], -float("inf"))
        return masked.argmax(dim=1)

    table = confuser_map.to(labels.device)[labels]
    valid = table.ge(0)
    safe_table = table.clamp_min(0)
    scores = source_logits.float().gather(1, safe_table)
    scores = scores.masked_fill(~valid, -float("inf"))
    selected = safe_table.gather(
        1,
        scores.argmax(dim=1, keepdim=True),
    ).squeeze(1)

    no_valid = ~valid.any(dim=1)
    if no_valid.any():
        masked = source_logits[no_valid].float().clone()
        masked.scatter_(
            1,
            labels[no_valid, None],
            -float("inf"),
        )
        selected[no_valid] = masked.argmax(dim=1)
    return selected



def choose_oof_confuser(
    labels: torch.Tensor,
    confuser_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the first ranked training-only OOF confuser for each label.

    In the revised all-class ENTO–MVL model every class is present in the
    decision space, so no class-subset membership test is needed. The
    returned validity mask is False only when a class has no stored OOF
    confuser.
    """
    if confuser_map is None:
        raise ValueError(
            "A training-only OOF confuser map is required for this objective."
        )

    table = confuser_map.to(labels.device)[labels]
    valid_entries = table.ge(0)
    valid = valid_entries.any(dim=1)
    first_valid_pos = valid_entries.float().argmax(dim=1)
    safe_table = table.clamp_min(0)
    confuser_class = safe_table.gather(
        1, first_valid_pos[:, None]
    ).squeeze(1)
    confuser_class = torch.where(
        valid, confuser_class, torch.full_like(confuser_class, -1)
    )
    return confuser_class, valid


def exact_mcnemar_p(corrected: int, degraded: int) -> float:
    import math

    discordant = int(corrected) + int(degraded)
    if discordant == 0:
        return 1.0
    smaller = min(int(corrected), int(degraded))
    tail = sum(
        math.comb(discordant, k)
        for k in range(smaller + 1)
    ) / float(2 ** discordant)
    return min(1.0, 2.0 * tail)
