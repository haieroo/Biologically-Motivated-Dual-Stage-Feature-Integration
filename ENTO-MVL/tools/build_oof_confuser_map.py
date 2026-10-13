from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def find_column(fieldnames, candidates):
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(
        f"Could not find any of {candidates}. Available: {fieldnames}"
    )


def main():
    parser = argparse.ArgumentParser(
        "Build a class-level confuser map from leakage-free OOF predictions."
    )
    parser.add_argument("--oof_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()

    source = Path(args.oof_csv)
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("OOF CSV has no header.")
        label_col = find_column(
            reader.fieldnames,
            ["label", "true_label", "target", "class_index"],
        )
        prediction_col = find_column(
            reader.fieldnames,
            ["source_prediction", "mvl_prediction", "prediction"],
        )
        fold_col = find_column(reader.fieldnames, ["fold"])
        counts = defaultdict(Counter)
        fold_values = set()
        total_rows = 0
        error_rows = 0
        for row in reader:
            label = int(row[label_col])
            prediction = int(row[prediction_col])
            fold_values.add(int(row[fold_col]))
            total_rows += 1
            if prediction != label:
                counts[label][prediction] += 1
                error_rows += 1

    if len(fold_values) < 2:
        raise RuntimeError(
            "OOF CSV contains fewer than two folds and is not acceptable."
        )

    mapping = {}
    classes_with_errors = 0
    for class_index in range(args.num_classes):
        neighbors = [
            int(class_id)
            for class_id, _ in counts[class_index].most_common(args.neighbors)
        ]
        mapping[str(class_index)] = neighbors
        classes_with_errors += int(bool(neighbors))

    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "oof_csv": str(source),
        "output_json": str(destination),
        "rows": total_rows,
        "oof_errors": error_rows,
        "oof_error_rate": error_rows / max(total_rows, 1),
        "folds": sorted(fold_values),
        "neighbors": args.neighbors,
        "classes_with_oof_errors": classes_with_errors,
        "online_hard_negative_fallback": False,
    }
    destination.with_suffix('.summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
