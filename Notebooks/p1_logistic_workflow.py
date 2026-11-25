from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil
import tempfile
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import validation_models


C_VALUES = (0.1, 1.0, 10.0)
CHUNK_ROWS = 20_000
LABEL_TO_CODE = {"Normal": 0, "Attack": 1}


def _context(validation_run_dir: Path) -> dict:
    return validation_models._load_context(
        validation_run_dir, "P1", current_p1_run=True
    )


def current_validation_run_dir(source_run_dir: Path | None = None) -> Path:
    return validation_models.latest_p1_validation_run_dir(source_run_dir)


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    solver="saga",
                    C=c_value,
                    max_iter=200,
                    tol=1e-3,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


def _fit(pipeline: Pipeline, x: np.ndarray, y: np.ndarray) -> float:
    estimated_peak = int(x.nbytes * 4)
    available = psutil.virtual_memory().available
    print(
        f"Peak RAM ước tính: {estimated_peak / 1024**3:.2f} GiB; "
        f"RAM khả dụng: {available / 1024**3:.2f} GiB."
    )
    if estimated_peak > available * 0.70:
        raise MemoryError("Không đủ RAM để fit Logistic Regression an toàn.")
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x, y)
    return time.perf_counter() - started


def _evaluate(pipeline: Pipeline, path: Path, context: dict, partition: str) -> dict:
    expected_rows = int(context["manifest"]["partition_rows"][partition])
    expected_counts = context["manifest"]["partition_label_counts"][partition]
    confusion = np.zeros((2, 2), dtype=np.int64)
    observed = np.zeros(2, dtype=np.int64)
    dtypes = validation_models._source_dtypes(context)
    for chunk in pd.read_csv(
        path, compression="gzip", chunksize=CHUNK_ROWS, dtype=dtypes,
        float_precision=context["manifest"].get("split_csv_float_precision"),
    ):
        labels = validation_models._validate_chunk(chunk, context)
        values = chunk[context["feature_columns"]].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Chuyển feature sang float32 tạo giá trị không hữu hạn.")
        predictions = pipeline.predict(values)
        confusion += confusion_matrix(labels, predictions, labels=[0, 1])
        observed += np.bincount(labels, minlength=2)
    expected = [int(expected_counts["Normal"]), int(expected_counts["Attack"])]
    if observed.tolist() != expected or int(observed.sum()) != expected_rows:
        raise ValueError(f"Số dòng hoặc nhãn của {partition} không khớp audit.")
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    if tn + fp == 0 or tp + fn == 0:
        raise ValueError(f"{partition} phải có cả Normal và Attack.")
    attack_precision = tp / (tp + fp) if tp + fp else 0.0
    attack_recall = tp / (tp + fn)
    attack_f1 = 2 * tp / (2 * tp + fp + fn)
    normal_f1 = 2 * tn / (2 * tn + fp + fn)
    return {
        "rows": expected_rows,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": (tn + tp) / expected_rows,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "attack_f1": attack_f1,
        "macro_f1": (normal_f1 + attack_f1) / 2,
        "fpr": fp / (tn + fp),
        "fnr": fn / (tp + fn),
    }


def prepare_validation_cache(validation_run_dir: Path) -> dict:
    return validation_models.prepare_training_cache(
        validation_run_dir, "P1", current_p1_run=True
    )


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = _context(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "logistic_validation_selection.json"
    if output.exists():
        raise FileExistsError(f"Kết quả lựa chọn đã tồn tại: {output}")
    cache_dir = run_dir / "training_cache"
    x = np.load(cache_dir / "X_train_internal_float32.npy", mmap_mode="r")
    y = np.load(cache_dir / "y_train_internal_uint8.npy", mmap_mode="r")
    rows = int(context["manifest"]["partition_rows"]["TrainInternal"])
    if x.shape != (rows, len(context["feature_columns"])) or y.shape != (rows,):
        raise ValueError("Cache Train nội bộ không khớp manifest.")
    results = []
    try:
        for c_value in C_VALUES:
            print(f"Đang fit và đánh giá Validation: C={c_value}")
            pipeline = _pipeline(c_value)
            fit_seconds = _fit(pipeline, x, y)
            metrics = _evaluate(pipeline, context["validation_path"], context, "Validation")
            results.append(
                {
                    "C": c_value,
                    "fit_seconds": fit_seconds,
                    "n_iter": int(pipeline.named_steps["logistic_regression"].n_iter_[0]),
                    **metrics,
                }
            )
            del pipeline
            gc.collect()
    finally:
        del x, y
        gc.collect()
    selected = max(results, key=lambda row: (row["macro_f1"], -row["C"]))
    summary = {
        "source_run_id": context["manifest"]["source_run_id"],
        "validation_run_id": context["manifest"]["validation_run_id"],
        "selected_C": selected["C"],
        "selection_rule": "highest macro_f1; smallest C on exact tie",
        "test_read_or_used": False,
        "results": results,
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"C được chọn theo Macro-F1: {selected['C']}")
    return pd.DataFrame(results)


def _selected_context(validation_run_dir: Path) -> tuple[dict, dict]:
    context = _context(validation_run_dir)
    selection_path = context["run_dir"] / "logistic_validation_selection.json"
    with selection_path.open(encoding="utf-8") as file:
        selection = json.load(file)
    manifest = context["manifest"]
    if (
        selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("test_read_or_used") is not False
        or selection.get("selected_C") not in C_VALUES
    ):
        raise ValueError("Lựa chọn Logistic không khớp run Validation.")
    return context, selection


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _selected_context(validation_run_dir)
    manifest = context["manifest"]
    source_dir = context["run_dir"].resolve().parent.parent
    train_path = source_dir / "train.csv.gz"
    output = context["run_dir"] / "logistic_selected_pipeline.joblib"
    summary_path = context["run_dir"] / "logistic_selected_training.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary đã tồn tại; không ghi đè.")
    if not train_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy Train gốc: {train_path}")
    rows = int(manifest["source_train_rows"])
    counts = manifest["source_train_label_counts"]
    matrix_bytes = rows * len(context["feature_columns"]) * np.dtype(np.float32).itemsize
    if shutil.disk_usage(context["run_dir"]).free < matrix_bytes + rows + 64 * 1024**2:
        raise OSError("Không đủ đĩa cho cache Train gốc.")
    with tempfile.TemporaryDirectory(prefix=".logistic_fit_", dir=context["run_dir"]) as temp:
        x = None
        y = None
        try:
            x = np.lib.format.open_memmap(
                Path(temp) / "x.npy", mode="w+", dtype=np.float32,
                shape=(rows, len(context["feature_columns"])),
            )
            y = np.lib.format.open_memmap(
                Path(temp) / "y.npy", mode="w+", dtype=np.uint8, shape=(rows,),
            )
            observed = np.zeros(2, dtype=np.int64)
            position = 0
            for chunk in pd.read_csv(
                train_path, compression="gzip", chunksize=CHUNK_ROWS,
                dtype=validation_models._source_dtypes(context),
                float_precision=manifest.get("split_csv_float_precision"),
            ):
                labels = validation_models._validate_chunk(chunk, context)
                stop = position + len(chunk)
                if stop > rows:
                    raise ValueError("Train gốc có nhiều dòng hơn manifest.")
                values = chunk[context["feature_columns"]].to_numpy(dtype=np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise ValueError("Chuyển feature sang float32 tạo giá trị không hữu hạn.")
                x[position:stop] = values
                y[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != rows or observed.tolist() != [counts["Normal"], counts["Attack"]]:
                raise ValueError("Số dòng hoặc nhãn Train gốc không khớp audit.")
            pipeline = _pipeline(float(selection["selected_C"]))
            fit_seconds = _fit(pipeline, x, y)
        finally:
            if x is not None:
                x._mmap.close()
            if y is not None:
                y._mmap.close()
        temporary_model = Path(temp) / "pipeline.joblib"
        joblib.dump(pipeline, temporary_model, compress=3)
        summary = {
            "source_run_id": manifest["source_run_id"],
            "validation_run_id": manifest["validation_run_id"],
            "selected_C": selection["selected_C"],
            "feature_columns": context["feature_columns"],
            "train_rows": rows,
            "fit_seconds": fit_seconds,
            "n_iter": int(pipeline.named_steps["logistic_regression"].n_iter_[0]),
            "test_read_or_used": False,
        }
        temporary_summary = Path(temp) / "training.json"
        with temporary_summary.open("x", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        temporary_model.replace(output)
        try:
            temporary_summary.replace(summary_path)
        except OSError:
            output.unlink()
            raise
    print(f"Đã lưu pipeline Logistic: {output}")
    return output


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _selected_context(validation_run_dir)
    output = context["run_dir"] / "logistic_test_metrics.json"
    model_path = context["run_dir"] / "logistic_selected_pipeline.joblib"
    summary_path = context["run_dir"] / "logistic_selected_training.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    if (
        summary.get("source_run_id") != context["manifest"]["source_run_id"]
        or summary.get("validation_run_id") != context["manifest"]["validation_run_id"]
        or summary.get("selected_C") != selection["selected_C"]
        or summary.get("feature_columns") != context["feature_columns"]
    ):
        raise ValueError("Model và cấu hình đã chọn không khớp.")
    source_dir = context["run_dir"].resolve().parent.parent
    with (source_dir / "split_manifest.json").open(encoding="utf-8") as file:
        split_manifest = json.load(file)
    if split_manifest.get("run_id") != context["manifest"]["source_run_id"]:
        raise ValueError("Manifest Test không khớp run nguồn.")
    counts = split_manifest["split_counts"]["Test"]
    test_context = {
        **context,
        "manifest": {
            **context["manifest"],
            "partition_rows": {"Test": counts["rows"]},
            "partition_label_counts": {"Test": {label: counts[label] for label in LABEL_TO_CODE}},
        },
    }
    metrics = _evaluate(joblib.load(model_path), source_dir / "test.csv.gz", test_context, "Test")
    with output.open("x", encoding="utf-8") as file:
        json.dump(
            {"source_run_id": summary["source_run_id"], "selected_C": summary["selected_C"],
             "previously_seen_test": True, "metrics": metrics},
            file, ensure_ascii=False, indent=2,
        )
    return pd.DataFrame([metrics])
