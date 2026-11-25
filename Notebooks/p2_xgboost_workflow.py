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

import p2_logistic_workflow as p2_data


CHUNK_ROWS = 20_000
LABELS = ("Normal", "Attack")
TRAIN_DAYS = {"Monday", "Tuesday", "Wednesday"}
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


def latest_validation_run_dir(required_artifacts: tuple[str, ...] = ()) -> Path:
    return p2_data.latest_validation_run_dir(required_artifacts=required_artifacts)


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


def _read_selection(validation_run_dir: Path) -> tuple[dict, dict]:
    context = p2_data.load_context(validation_run_dir)
    path = context["run_dir"] / "p2_xgboost_validation_selection.json"
    with path.open(encoding="utf-8") as file:
        selection = json.load(file)
    manifest = context["validation_manifest"]
    if (
        selection.get("protocol") != "P2"
        or selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("xgboost_version") != xgboost.__version__
        or selection.get("early_stopping") is not False
        or selection.get("test_read_or_used") is not False
        or selection.get("test_features_used_to_filter_train") is not True
        or selection.get("selected_parameters")
        not in [_parameters(candidate) for candidate in CANDIDATES]
    ):
        raise ValueError("Lựa chọn XGBoost không khớp run hoặc môi trường P2.")
    return context, selection


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = p2_data.load_context(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "p2_xgboost_validation_selection.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Validation đã tồn tại: {output}")
    _check_resources(run_dir)
    context = p2_data.prepare_validation_cache(validation_run_dir)
    cache_dir = run_dir / "p2_logistic_training_cache"
    x = np.load(cache_dir / "X_train_internal_float32.npy", mmap_mode="r")
    y = np.load(cache_dir / "y_train_internal_uint8.npy", mmap_mode="r")
    manifest = context["validation_manifest"]
    rows = int(manifest["partition_rows"]["TrainInternal"])
    counts = manifest["partition_label_counts"]["TrainInternal"]
    try:
        if (
            x.shape != (rows, len(context["feature_columns"]))
            or x.dtype != np.dtype(np.float32)
            or y.shape != (rows,)
            or y.dtype != np.dtype(np.uint8)
            or np.bincount(y, minlength=2).tolist() != [int(counts[label]) for label in LABELS]
        ):
            raise ValueError("Cache TrainInternal không khớp manifest P2.")
        results = []
        for candidate in CANDIDATES:
            print(f"Đang fit XGBoost P2 trên TrainInternal: {candidate}")
            model = XGBClassifier(**_parameters(candidate))
            fit_seconds = _fit(model, x, y, run_dir)
            metrics = p2_data.evaluate_partition(model, run_dir, "Validation")
            results.append({**candidate, "fit_seconds": fit_seconds, **metrics})
            del model
            gc.collect()
    finally:
        x._mmap.close()
        y._mmap.close()
        del x, y
        gc.collect()

    selected = max(
        results,
        key=lambda row: (
            row["macro_f1"], -row["max_depth"],
            -row["n_estimators"], row["min_child_weight"],
        ),
    )
    selected_parameters = _parameters({key: selected[key] for key in CANDIDATES[0]})
    _write_json_new(output, {
        "protocol": "P2",
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
        "test_features_used_to_filter_train": True,
        "results": results,
    })
    print(f"XGBoost P2 được chọn theo Macro-F1: {selected_parameters}")
    return pd.DataFrame(results)


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    train_path = context["paths"]["train"]
    model_path = run_dir / "p2_xgboost_selected_model.joblib"
    summary_path = run_dir / "p2_xgboost_selected_training.json"
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary XGBoost P2 đã tồn tại.")
    _check_resources(run_dir)
    counts = context["split_manifest"]["counts"]["Train"]
    rows = sum(int(counts[label]) for label in LABELS)
    features = context["feature_columns"]
    signature = p2_data.file_signature(train_path)
    model = XGBClassifier(**selection["selected_parameters"])
    with tempfile.TemporaryDirectory(prefix=".p2_xgboost_fit_", dir=run_dir) as temp:
        temp_dir = Path(temp)
        x = y = None
        try:
            x = np.lib.format.open_memmap(
                temp_dir / "x_train.npy", mode="w+", dtype=np.float32,
                shape=(rows, len(features)),
            )
            y = np.lib.format.open_memmap(
                temp_dir / "y_train.npy", mode="w+", dtype=np.uint8, shape=(rows,),
            )
            observed = np.zeros(2, dtype=np.int64)
            position = 0
            for chunk in pd.read_csv(
                train_path, compression="gzip", chunksize=CHUNK_ROWS,
                dtype=p2_data.data_dtypes(context), float_precision="round_trip",
            ):
                labels = p2_data.validate_chunk(chunk, context, TRAIN_DAYS)
                stop = position + len(chunk)
                if stop > rows:
                    raise ValueError("Train P2 có nhiều dòng hơn split manifest.")
                values = chunk[features].to_numpy(dtype=np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise ValueError("Train P2 có feature không hữu hạn sau đổi float32.")
                x[position:stop] = values
                y[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != rows or observed.tolist() != [int(counts[label]) for label in LABELS]:
                raise ValueError("Train P2 không khớp số dòng/nhãn manifest.")
            p2_data.check_file_signature(train_path, signature)
            fit_seconds = _fit(model, x, y, run_dir)
        finally:
            if x is not None:
                x._mmap.close()
            if y is not None:
                y._mmap.close()

        temp_model = temp_dir / "model.joblib"
        joblib.dump(model, temp_model, compress=3)
        summary = {
            "protocol": "P2",
            "source_run_id": context["validation_manifest"]["source_run_id"],
            "validation_run_id": context["validation_manifest"]["validation_run_id"],
            "fit_partition": "Train",
            "train_rows": rows,
            "train_label_counts": counts,
            "feature_columns": features,
            "selected_parameters": selection["selected_parameters"],
            "xgboost_version": xgboost.__version__,
            "fit_seconds": fit_seconds,
            "n_estimators_observed": model.get_booster().num_boosted_rounds(),
            "early_stopping": False,
            "test_read_or_used": False,
            "test_features_used_to_filter_train": True,
        }
        with temp_model.open("rb") as source, model_path.open("xb") as target:
            try:
                shutil.copyfileobj(source, target)
            except BaseException:
                target.close()
                model_path.unlink()
                raise
        try:
            _write_json_new(summary_path, summary)
        except BaseException:
            model_path.unlink()
            raise
    print(f"Đã lưu XGBoost P2: {model_path}")
    return model_path


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    model_path = run_dir / "p2_xgboost_selected_model.joblib"
    summary_path = run_dir / "p2_xgboost_selected_training.json"
    output = run_dir / "p2_xgboost_test_metrics.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    if not model_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Fit XGBoost trên Train P2 trước khi đánh giá Test.")
    with summary_path.open(encoding="utf-8") as file:
        training = json.load(file)
    manifest = context["validation_manifest"]
    if (
        training.get("protocol") != "P2"
        or training.get("source_run_id") != manifest["source_run_id"]
        or training.get("validation_run_id") != manifest["validation_run_id"]
        or training.get("fit_partition") != "Train"
        or training.get("selected_parameters") != selection["selected_parameters"]
        or training.get("feature_columns") != context["feature_columns"]
        or training.get("train_rows")
        != sum(int(context["split_manifest"]["counts"]["Train"][label]) for label in LABELS)
        or training.get("xgboost_version") != xgboost.__version__
        or training.get("early_stopping") is not False
        or training.get("test_read_or_used") is not False
        or training.get("test_features_used_to_filter_train") is not True
    ):
        raise ValueError("Model XGBoost đã fit không khớp run/cấu hình P2.")
    model = joblib.load(model_path)
    if (
        not isinstance(model, XGBClassifier)
        or list(model.classes_) != [0, 1]
        or model.n_features_in_ != len(context["feature_columns"])
        or model.get_booster().num_boosted_rounds()
        != selection["selected_parameters"]["n_estimators"]
    ):
        raise ValueError("Model XGBoost không khớp nhãn, feature hoặc số cây.")
    metrics = p2_data.evaluate_partition(model, run_dir, "Test")
    _write_json_new(output, {
        "protocol": "P2",
        "source_run_id": manifest["source_run_id"],
        "validation_run_id": manifest["validation_run_id"],
        "selected_parameters": selection["selected_parameters"],
        "xgboost_version": xgboost.__version__,
        "partition": "Test",
        "previously_seen_test": True,
        "test_read_or_used": True,
        "test_features_used_to_filter_train": True,
        "metrics": metrics,
    })
    return pd.DataFrame([metrics])
