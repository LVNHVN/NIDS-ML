from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import psutil
import xgboost
from xgboost import XGBClassifier

import validation_models


CHUNK_ROWS = 20_000
LABELS = ("Normal", "Attack")
FIXED_PARAMETERS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cpu",
    "max_bin": 128,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1,
    "scale_pos_weight": 1,
    "random_state": 42,
    "n_jobs": 4,
}
CANDIDATES = (
    {"max_depth": 4, "n_estimators": 150, "min_child_weight": 1},
    {"max_depth": 6, "n_estimators": 150, "min_child_weight": 1},
    {"max_depth": 8, "n_estimators": 150, "min_child_weight": 1},
    {"max_depth": 6, "n_estimators": 250, "min_child_weight": 1},
    {"max_depth": 6, "n_estimators": 150, "min_child_weight": 5},
)
MIN_AVAILABLE_RAM = 4 * 1024**3
MIN_FREE_DISK = 3 * 1024**3


def current_validation_run_dir(source_run_dir: Path | None = None) -> Path:
    return validation_models.latest_p1_validation_run_dir(source_run_dir)


def _context(validation_run_dir: Path) -> dict:
    return validation_models._load_context(
        validation_run_dir, "P1", current_p1_run=True
    )


def _parameters(candidate: dict) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Cấu hình XGBoost không hợp lệ: {candidate}")
    return {**FIXED_PARAMETERS, **candidate}


def _check_resources(run_dir: Path) -> None:
    available = psutil.virtual_memory().available
    free = shutil.disk_usage(run_dir).free
    print(
        f"RAM khả dụng: {available / 1024**3:.2f} GiB; "
        f"đĩa trống: {free / 1024**3:.2f} GiB. "
        "Đây không phải giới hạn peak RAM thực tế."
    )
    if available < MIN_AVAILABLE_RAM:
        raise MemoryError("Cần ít nhất 4 GiB RAM khả dụng trước khi fit XGBoost.")
    if free < MIN_FREE_DISK:
        raise OSError("Cần ít nhất 3 GiB đĩa trống trước khi fit XGBoost.")


def _write_json_new(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as file:
        try:
            file.write(text)
        except BaseException:
            file.close()
            path.unlink()
            raise


def _fit(model: XGBClassifier, x: np.ndarray, y: np.ndarray, run_dir: Path) -> float:
    _check_resources(run_dir)
    started = time.perf_counter()
    model.fit(x, y)
    return time.perf_counter() - started


def prepare_validation_cache(validation_run_dir: Path) -> dict:
    context = _context(validation_run_dir)
    if (context["run_dir"] / "xgboost_validation_selection.json").exists():
        raise FileExistsError("Kết quả lựa chọn XGBoost đã tồn tại; không ghi đè.")
    _check_resources(context["run_dir"])
    return validation_models.prepare_training_cache(
        validation_run_dir, "P1", current_p1_run=True
    )


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = _context(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "xgboost_validation_selection.json"
    if output.exists():
        raise FileExistsError(f"Kết quả lựa chọn đã tồn tại: {output}")

    cache_dir = run_dir / "training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError("Chạy prepare_validation_cache trước khi so sánh.")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    manifest = context["manifest"]
    rows = int(manifest["partition_rows"]["TrainInternal"])
    counts = manifest["partition_label_counts"]["TrainInternal"]
    if (
        x.shape != (rows, len(context["feature_columns"]))
        or x.dtype != np.dtype(np.float32)
        or y.shape != (rows,)
        or y.dtype != np.dtype(np.uint8)
        or np.bincount(y, minlength=2).tolist() != [int(counts[label]) for label in LABELS]
    ):
        raise ValueError("Cache Train nội bộ không khớp manifest.")

    results = []
    try:
        for candidate in CANDIDATES:
            print(f"Đang fit và đánh giá Validation: {candidate}")
            model = XGBClassifier(**_parameters(candidate))
            fit_seconds = _fit(model, x, y, run_dir)
            metrics = validation_models.evaluate_binary_classifier(
                model, context["validation_path"], context, "Validation"
            )
            results.append({**candidate, "fit_seconds": fit_seconds, **metrics})
            del model
            gc.collect()
    finally:
        del x, y
        gc.collect()

    selected = max(
        results,
        key=lambda row: (
            row["macro_f1"],
            -row["max_depth"],
            -row["n_estimators"],
            row["min_child_weight"],
        ),
    )
    selected_parameters = _parameters({key: selected[key] for key in CANDIDATES[0]})
    _write_json_new(output, {
        "source_run_id": manifest["source_run_id"],
        "validation_run_id": manifest["validation_run_id"],
        "xgboost_version": xgboost.__version__,
        "selected_parameters": selected_parameters,
        "selection_rule": (
            "highest macro_f1; on exact tie, lower max_depth, "
            "fewer n_estimators, then higher min_child_weight"
        ),
        "early_stopping": False,
        "test_read_or_used": False,
        "results": results,
    })
    print(f"Cấu hình được chọn theo Macro-F1: {selected_parameters}")
    return pd.DataFrame(results)


def _selected_context(validation_run_dir: Path) -> tuple[dict, dict]:
    context = _context(validation_run_dir)
    with (context["run_dir"] / "xgboost_validation_selection.json").open(
        encoding="utf-8"
    ) as file:
        selection = json.load(file)
    manifest = context["manifest"]
    if (
        selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("test_read_or_used") is not False
        or selection.get("early_stopping") is not False
        or selection.get("xgboost_version") != xgboost.__version__
        or selection.get("selected_parameters") not in [
            _parameters(candidate) for candidate in CANDIDATES
        ]
    ):
        raise ValueError("Lựa chọn XGBoost không khớp run hoặc môi trường.")
    return context, selection


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _selected_context(validation_run_dir)
    manifest = context["manifest"]
    source_dir = context["run_dir"].resolve().parent.parent
    train_path = source_dir / "train.csv.gz"
    output = context["run_dir"] / "xgboost_selected_model.joblib"
    summary_path = context["run_dir"] / "xgboost_selected_training.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary đã tồn tại; không ghi đè.")
    if not train_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy Train gốc: {train_path}")
    with (source_dir / "split_manifest.json").open(encoding="utf-8") as file:
        split_manifest = json.load(file)
    counts = manifest["source_train_label_counts"]
    train_counts = split_manifest["split_counts"]["Train"]
    rows = int(manifest["source_train_rows"])
    if (
        split_manifest.get("run_id") != manifest["source_run_id"]
        or split_manifest.get("feature_columns") != context["feature_columns"]
        or split_manifest.get("split_csv_float_precision") != "round_trip"
        or int(train_counts["rows"]) != rows
        or any(int(train_counts[label]) != int(counts[label]) for label in LABELS)
    ):
        raise ValueError("Manifest Train gốc không khớp Validation P1.")
    _check_resources(context["run_dir"])

    model = XGBClassifier(**selection["selected_parameters"])
    with tempfile.TemporaryDirectory(
        prefix=".xgboost_fit_", dir=context["run_dir"]
    ) as temp:
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
                values = chunk[context["feature_columns"]].to_numpy(
                    dtype=np.float32, copy=False
                )
                if not np.isfinite(values).all():
                    raise ValueError("Feature không hữu hạn sau chuyển float32.")
                x[position:stop] = values
                y[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != rows or observed.tolist() != [
                int(counts[label]) for label in LABELS
            ]:
                raise ValueError("Số dòng hoặc nhãn Train gốc không khớp audit.")
            fit_seconds = _fit(model, x, y, context["run_dir"])
        finally:
            if x is not None:
                x._mmap.close()
            if y is not None:
                y._mmap.close()

        temporary_model = Path(temp) / "model.joblib"
        joblib.dump(model, temporary_model, compress=3)
        summary = {
            "source_run_id": manifest["source_run_id"],
            "validation_run_id": manifest["validation_run_id"],
            "fit_partition": "Train",
            "train_rows": rows,
            "feature_columns": context["feature_columns"],
            "selected_parameters": selection["selected_parameters"],
            "xgboost_version": xgboost.__version__,
            "fit_seconds": fit_seconds,
            "n_estimators_observed": model.get_booster().num_boosted_rounds(),
            "early_stopping": False,
            "test_read_or_used": False,
        }
        with temporary_model.open("rb") as source, output.open("xb") as target:
            try:
                shutil.copyfileobj(source, target)
            except BaseException:
                target.close()
                output.unlink()
                raise
        try:
            _write_json_new(summary_path, summary)
        except BaseException:
            output.unlink()
            raise
    print(f"Đã lưu XGBoost: {output}")
    return output


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _selected_context(validation_run_dir)
    output = context["run_dir"] / "xgboost_test_metrics.json"
    model_path = context["run_dir"] / "xgboost_selected_model.joblib"
    summary_path = context["run_dir"] / "xgboost_selected_training.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    if not model_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Fit XGBoost trên Train trước khi đánh giá Test.")
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    manifest = context["manifest"]
    if (
        summary.get("source_run_id") != manifest["source_run_id"]
        or summary.get("validation_run_id") != manifest["validation_run_id"]
        or summary.get("selected_parameters") != selection["selected_parameters"]
        or summary.get("feature_columns") != context["feature_columns"]
        or summary.get("fit_partition") != "Train"
        or summary.get("train_rows") != manifest["source_train_rows"]
        or summary.get("xgboost_version") != xgboost.__version__
        or summary.get("early_stopping") is not False
        or summary.get("test_read_or_used") is not False
    ):
        raise ValueError("Model đã fit không khớp cấu hình, run hoặc môi trường.")
    source_dir = context["run_dir"].resolve().parent.parent
    with (source_dir / "split_manifest.json").open(encoding="utf-8") as file:
        split_manifest = json.load(file)
    if (
        split_manifest.get("run_id") != manifest["source_run_id"]
        or split_manifest.get("feature_columns") != context["feature_columns"]
        or split_manifest.get("split_csv_float_precision") != "round_trip"
    ):
        raise ValueError("Manifest Test không khớp run nguồn.")
    counts = split_manifest["split_counts"]["Test"]
    test_path = source_dir / "test.csv.gz"
    if not test_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy Test P1: {test_path}")
    test_context = {
        **context,
        "manifest": {
            **manifest,
            "partition_rows": {"Test": int(counts["rows"])},
            "partition_label_counts": {
                "Test": {label: int(counts[label]) for label in LABELS}
            },
        },
    }
    model = joblib.load(model_path)
    if (
        not isinstance(model, XGBClassifier)
        or list(model.classes_) != [0, 1]
        or model.n_features_in_ != len(context["feature_columns"])
        or model.get_booster().num_boosted_rounds()
        != selection["selected_parameters"]["n_estimators"]
    ):
        raise ValueError("Model XGBoost không khớp nhãn, feature hoặc số cây.")
    metrics = validation_models.evaluate_binary_classifier(
        model, test_path, test_context, "Test"
    )
    _write_json_new(output, {
        "source_run_id": summary["source_run_id"],
        "validation_run_id": summary["validation_run_id"],
        "selected_parameters": summary["selected_parameters"],
        "xgboost_version": xgboost.__version__,
        "partition": "Test",
        "test_read_or_used": True,
        "metrics": metrics,
    })
    return pd.DataFrame([metrics])
