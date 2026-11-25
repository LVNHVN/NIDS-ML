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
from sklearn.ensemble import RandomForestClassifier

import p2_logistic_workflow as p2_data


CHUNK_ROWS = 20_000
LABELS = ("Normal", "Attack")
TRAIN_DAYS = {"Monday", "Tuesday", "Wednesday"}
FIXED_PARAMETERS = {
    "criterion": "gini",
    "max_leaf_nodes": 10_000,
    "n_estimators": 50,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": 0.25,
    "n_jobs": 1,
    "class_weight": None,
    "random_state": 42,
}
CANDIDATES = (
    {"max_depth": 12, "min_samples_leaf": 20},
    {"max_depth": 16, "min_samples_leaf": 20},
    {"max_depth": 20, "min_samples_leaf": 20},
    {"max_depth": 20, "min_samples_leaf": 10},
    {"max_depth": 20, "min_samples_leaf": 50},
)


def latest_validation_run_dir(required_artifacts: tuple[str, ...] = ()) -> Path:
    return p2_data.latest_validation_run_dir(required_artifacts=required_artifacts)


def _parameters(candidate: dict) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown Random Forest candidate: {candidate}")
    return {**FIXED_PARAMETERS, **candidate}


def _check_fit_memory(matrix_bytes: int) -> None:
    node_bytes = FIXED_PARAMETERS["n_estimators"] * (
        2 * FIXED_PARAMETERS["max_leaf_nodes"] - 1
    ) * 128
    estimated_peak = int(matrix_bytes * 3 + node_bytes + 512 * 1024**2)
    available = psutil.virtual_memory().available
    print(
        f"Peak RAM ước tính: {estimated_peak / 1024**3:.2f} GiB; "
        f"RAM khả dụng: {available / 1024**3:.2f} GiB."
    )
    if estimated_peak > available * 0.70:
        raise MemoryError("Không đủ RAM để fit Random Forest P2 an toàn.")


def _fit(model: RandomForestClassifier, x: np.ndarray, y: np.ndarray) -> float:
    _check_fit_memory(x.nbytes)
    started = time.perf_counter()
    model.fit(x, y)
    return time.perf_counter() - started


def _read_selection(validation_run_dir: Path) -> tuple[dict, dict]:
    context = p2_data.load_context(validation_run_dir)
    path = context["run_dir"] / "p2_random_forest_validation_selection.json"
    with path.open(encoding="utf-8") as file:
        selection = json.load(file)
    manifest = context["validation_manifest"]
    if (
        selection.get("protocol") != "P2"
        or selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("test_read_or_used") is not False
        or selection.get("test_features_used_to_filter_train") is not True
        or selection.get("selected_parameters")
        not in [_parameters(candidate) for candidate in CANDIDATES]
    ):
        raise ValueError("Lựa chọn Random Forest không khớp Validation P2.")
    return context, selection


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = p2_data.load_context(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "p2_random_forest_validation_selection.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Validation đã tồn tại: {output}")
    rows = int(context["validation_manifest"]["partition_rows"]["TrainInternal"])
    _check_fit_memory(rows * len(context["feature_columns"]) * 4)
    context = p2_data.prepare_validation_cache(validation_run_dir)

    cache_dir = run_dir / "p2_logistic_training_cache"
    x = np.load(cache_dir / "X_train_internal_float32.npy", mmap_mode="r")
    y = np.load(cache_dir / "y_train_internal_uint8.npy", mmap_mode="r")
    counts = context["validation_manifest"]["partition_label_counts"]["TrainInternal"]
    if (
        x.shape != (rows, len(context["feature_columns"]))
        or x.dtype != np.dtype(np.float32)
        or y.shape != (rows,)
        or y.dtype != np.dtype(np.uint8)
        or np.bincount(y, minlength=2).tolist() != [int(counts[label]) for label in LABELS]
    ):
        raise ValueError("Cache TrainInternal không khớp manifest P2.")
    results = []
    try:
        for candidate in CANDIDATES:
            print(
                "Đang fit Random Forest P2 trên TrainInternal: "
                f"max_depth={candidate['max_depth']}, "
                f"min_samples_leaf={candidate['min_samples_leaf']}"
            )
            model = RandomForestClassifier(**_parameters(candidate))
            fit_seconds = _fit(model, x, y)
            metrics = p2_data.evaluate_partition(model, run_dir, "Validation")
            results.append({
                **_parameters(candidate),
                "fit_seconds": fit_seconds,
                "max_tree_depth_observed": int(max(tree.get_depth() for tree in model.estimators_)),
                "total_leaf_count": int(sum(tree.get_n_leaves() for tree in model.estimators_)),
                **metrics,
            })
            del model
            gc.collect()
    finally:
        del x, y
        gc.collect()

    selected = max(
        results, key=lambda row: (
            row["macro_f1"], -row["max_depth"], row["min_samples_leaf"]
        )
    )
    selected_parameters = _parameters({
        "max_depth": selected["max_depth"],
        "min_samples_leaf": selected["min_samples_leaf"],
    })
    summary = {
        "protocol": "P2",
        "source_run_id": context["validation_manifest"]["source_run_id"],
        "validation_run_id": context["validation_manifest"]["validation_run_id"],
        "selected_parameters": selected_parameters,
        "selection_rule": (
            "highest macro_f1; on exact tie, lower max_depth then higher min_samples_leaf"
        ),
        "test_read_or_used": False,
        "test_features_used_to_filter_train": True,
        "results": results,
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Random Forest P2 được chọn theo Macro-F1: {selected_parameters}")
    return pd.DataFrame(results)


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    train_path = context["paths"]["train"]
    model_path = run_dir / "p2_random_forest_selected_model.joblib"
    summary_path = run_dir / "p2_random_forest_selected_training.json"
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary Random Forest P2 đã tồn tại.")

    counts = context["split_manifest"]["counts"]["Train"]
    rows = sum(int(counts[label]) for label in LABELS)
    features = context["feature_columns"]
    matrix_bytes = rows * len(features) * np.dtype(np.float32).itemsize
    _check_fit_memory(matrix_bytes)
    model_budget = FIXED_PARAMETERS["n_estimators"] * (
        2 * FIXED_PARAMETERS["max_leaf_nodes"] - 1
    ) * 128
    if shutil.disk_usage(run_dir).free < matrix_bytes + rows + model_budget + 64 * 1024**2:
        raise OSError("Không đủ dung lượng đĩa cho fit Random Forest P2.")
    signature = p2_data.file_signature(train_path)
    model = RandomForestClassifier(**selection["selected_parameters"])
    with tempfile.TemporaryDirectory(prefix=".p2_random_forest_fit_", dir=run_dir) as temp:
        temp_dir = Path(temp)
        x = y = None
        try:
            x = np.lib.format.open_memmap(
                temp_dir / "x_train.npy", mode="w+", dtype=np.float32,
                shape=(rows, len(features)),
            )
            y = np.lib.format.open_memmap(
                temp_dir / "y_train.npy", mode="w+", dtype=np.uint8,
                shape=(rows,),
            )
            position = 0
            observed = np.zeros(2, dtype=np.int64)
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
            fit_seconds = _fit(model, x, y)
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
            "fit_seconds": fit_seconds,
            "n_estimators_observed": len(model.estimators_),
            "max_tree_depth_observed": int(max(tree.get_depth() for tree in model.estimators_)),
            "total_leaf_count": int(sum(tree.get_n_leaves() for tree in model.estimators_)),
            "test_read_or_used": False,
            "test_features_used_to_filter_train": True,
        }
        temp_summary = temp_dir / "training.json"
        with temp_summary.open("x", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        with temp_model.open("rb") as source, model_path.open("xb") as target:
            try:
                shutil.copyfileobj(source, target)
            except BaseException:
                target.close()
                model_path.unlink()
                raise
        try:
            with temp_summary.open("rb") as source, summary_path.open("xb") as target:
                try:
                    shutil.copyfileobj(source, target)
                except BaseException:
                    target.close()
                    summary_path.unlink()
                    raise
        except BaseException:
            model_path.unlink()
            raise
    print(f"Đã lưu Random Forest P2: {model_path}")
    return model_path


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    model_path = run_dir / "p2_random_forest_selected_model.joblib"
    summary_path = run_dir / "p2_random_forest_selected_training.json"
    output = run_dir / "p2_random_forest_test_metrics.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    if not model_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Fit Random Forest trên Train P2 trước khi đánh giá Test.")
    with summary_path.open(encoding="utf-8") as file:
        training = json.load(file)
    if (
        training.get("protocol") != "P2"
        or training.get("source_run_id") != context["validation_manifest"]["source_run_id"]
        or training.get("validation_run_id") != context["validation_manifest"]["validation_run_id"]
        or training.get("fit_partition") != "Train"
        or training.get("selected_parameters") != selection["selected_parameters"]
        or training.get("feature_columns") != context["feature_columns"]
        or training.get("train_rows")
        != sum(int(context["split_manifest"]["counts"]["Train"][label]) for label in LABELS)
        or training.get("test_read_or_used") is not False
        or training.get("test_features_used_to_filter_train") is not True
    ):
        raise ValueError("Model Random Forest đã fit không khớp run/cấu hình P2.")
    model = joblib.load(model_path)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Random Forest classes không khớp Normal=0/Attack=1.")
    metrics = p2_data.evaluate_partition(model, run_dir, "Test")
    result = {
        "protocol": "P2",
        "source_run_id": context["validation_manifest"]["source_run_id"],
        "validation_run_id": context["validation_manifest"]["validation_run_id"],
        "selected_parameters": selection["selected_parameters"],
        "partition": "Test",
        "previously_seen_test": True,
        "test_read_or_used": True,
        "test_features_used_to_filter_train": True,
        "metrics": metrics,
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
    return pd.DataFrame([metrics])
