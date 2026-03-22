from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

try:
    from ml.cnn_model.train_cnn import load_inference_artifacts, predict_ecg
except ImportError:
    from train_cnn import load_inference_artifacts, predict_ecg


SUPPORTED_SUFFIXES = {".hea", ".dat"}


def normalize_record_path(path: Path) -> str:
    # wfdb.rdsamp expects record path without .hea/.dat extension.
    if path.suffix.lower() in SUPPORTED_SUFFIXES:
        return str(path.with_suffix(""))
    return str(path)


def collect_records(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    pattern = "**/*.hea" if recursive else "*.hea"
    return sorted(input_path.glob(pattern))


def run_batch_predict(
    input_path: str,
    output_csv: str,
    threshold: float = 0.5,
    recursive: bool = True,
    top_k: int = 5,
) -> pd.DataFrame:
    records = collect_records(Path(input_path), recursive=recursive)
    model, label_names = load_inference_artifacts()

    rows = []
    for record in records:
        record_for_wfdb = normalize_record_path(record)
        try:
            preds = predict_ecg(record_for_wfdb, threshold=threshold, model=model, label_names=label_names)
            if top_k > 0:
                preds = preds[:top_k]

            rows.append(
                {
                    "record_path": str(record),
                    "n_predictions": len(preds),
                    "predictions_json": json.dumps(preds, ensure_ascii=True),
                }
            )
        except Exception as err:
            rows.append(
                {
                    "record_path": str(record),
                    "n_predictions": 0,
                    "predictions_json": "[]",
                    "error": str(err),
                }
            )

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ECG multi-label prediction for PTB-XL records")
    parser.add_argument("--input", required=True, help="Path to a single ECG file or directory containing .hea files")
    parser.add_argument("--output", default="ml/cnn_model/predictions.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction confidence threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Keep top-K predictions per record (0 means keep all)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recursively search input directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_batch_predict(
        input_path=args.input,
        output_csv=args.output,
        threshold=args.threshold,
        recursive=not args.no_recursive,
        top_k=args.top_k,
    )
    print(f"Saved {len(df)} prediction rows to: {args.output}")


if __name__ == "__main__":
    main()
