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
from sklearn.tree import DecisionTreeClassifier

import p2_logistic_workflow as p2_data


CHUNK_ROWS = 20_000
LABELS = ("Normal", "Attack")
TRAIN_DAYS = {"Monday", "Tuesday", "Wednesday"}
FIXED_PARAMETERS = {
    "criterion": "gini",
    "max_leaf_nodes": 50_000,
    "class_weight": None,
    "random_state": 42,
}
CANDIDATES = (
    {"max_depth": 12, "min_samples_leaf": 10},
    {"max_depth": 16, "min_samples_leaf": 10},
    {"max_depth": 20, "min_samples_leaf": 10},
    {"max_depth": 20, "min_samples_leaf": 25},
    {"max_depth": 20, "min_samples_leaf": 50},
)


def latest_validation_run_dir(
    required_artifacts: tuple[str, ...] = (),
) -> Path:
    return p2_data.latest_validation_run_dir(required_artifacts=required_artifacts)


def _parameters(candidate: dict) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown Decision Tree candidate: {candidate}")
    return {**FIXED_PARAMETERS, **candidate}


def _new_model(parameters: dict) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(**parameters)


def _fit(model: DecisionTreeClassifier, x_train: np.ndarray, y_train: np.ndarray) -> float:
    node_bytes = (2 * FIXED_PARAMETERS["max_leaf_nodes"] - 1) * 128
    estimated_peak = int(x_train.nbytes * 3 + node_bytes + 512 * 1024**2)
    available = psutil.virtual_memory().available
    print(
        f"Peak RAM ước tính: {estimated_peak / 1024**3:.2f} GiB; "
        f"RAM khả dụng: {available / 1024**3:.2f} GiB."
    )
    if estimated_peak > available * 0.70:
        raise MemoryError("Không đủ RAM để fit Decision Tree P2 an toàn.")
    started = time.perf_counter()
    model.fit(x_train, y_train)
    return time.perf_counter() - started


def _read_selection(validation_run_dir: Path) -> tuple[dict, dict]:
    context = p2_data.load_context(validation_run_dir)
    path = context["run_dir"] / "p2_decision_tree_validation_selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có kết quả chọn cấu hình: {path}")
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
        raise ValueError("Lựa chọn Decision Tree không khớp Validation P2.")
    return context, selection


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = p2_data.prepare_validation_cache(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "p2_decision_tree_validation_selection.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Validation đã tồn tại: {output}")

    cache_dir = run_dir / "p2_logistic_training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError("Cache TrainInternal P2 chưa hoàn chỉnh.")
    x_train = np.load(x_path, mmap_mode="r")
    y_train = np.load(y_path, mmap_mode="r")
    train_rows = int(context["validation_manifest"]["partition_rows"]["TrainInternal"])
    expected_counts = context["validation_manifest"]["partition_label_counts"]["TrainInternal"]
    if (
        x_train.shape != (train_rows, len(context["feature_columns"]))
        or x_train.dtype != np.dtype(np.float32)
        or y_train.shape != (train_rows,)
        or y_train.dtype != np.dtype(np.uint8)
        or np.bincount(y_train, minlength=2).tolist()
        != [int(expected_counts[label]) for label in LABELS]
    ):
        raise ValueError("Cache TrainInternal không khớp manifest P2.")

    results = []
    try:
        for candidate in CANDIDATES:
            parameters = _parameters(candidate)
            print(
                "Đang fit Decision Tree P2 trên TrainInternal: "
                f"max_depth={candidate['max_depth']}, "
                f"min_samples_leaf={candidate['min_samples_leaf']}"
            )
            model = _new_model(parameters)
            fit_seconds = _fit(model, x_train, y_train)
            metrics = p2_data.evaluate_partition(
                model, context["run_dir"], "Validation"
            )
            results.append({
                **parameters,
                "fit_seconds": fit_seconds,
                "observed_depth": int(model.get_depth()),
                "leaf_count": int(model.get_n_leaves()),
                **metrics,
            })
            del model
            gc.collect()
    finally:
        del x_train, y_train
        gc.collect()

    selected = max(
        results,
        key=lambda row: (
            row["macro_f1"],
            -row["max_depth"],
            row["min_samples_leaf"],
        ),
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
            "highest macro_f1; on exact tie, lower max_depth then "
            "higher min_samples_leaf"
        ),
        "test_read_or_used": False,
        "test_features_used_to_filter_train": True,
        "results": results,
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Decision Tree P2 được chọn theo Macro-F1: {selected_parameters}")
    return pd.DataFrame(results)


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    train_path = context["paths"]["train"]
    model_path = run_dir / "p2_decision_tree_selected_model.joblib"
    summary_path = run_dir / "p2_decision_tree_selected_training.json"
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary Decision Tree P2 đã tồn tại.")

    split_counts = context["split_manifest"]["counts"]["Train"]
    train_rows = sum(int(split_counts[label]) for label in LABELS)
    features = context["feature_columns"]
    matrix_bytes = train_rows * len(features) * np.dtype(np.float32).itemsize
    if shutil.disk_usage(run_dir).free < matrix_bytes + train_rows + 64 * 1024**2:
        raise OSError("Không đủ dung lượng đĩa cho fit Decision Tree P2.")
    signature = p2_data.file_signature(train_path)
    model = _new_model(selection["selected_parameters"])
    with tempfile.TemporaryDirectory(
        prefix=".p2_decision_tree_fit_", dir=run_dir
    ) as temp:
        temp_dir = Path(temp)
        x_map = y_map = None
        try:
            x_map = np.lib.format.open_memmap(
                temp_dir / "x_train.npy", mode="w+", dtype=np.float32,
                shape=(train_rows, len(features)),
            )
            y_map = np.lib.format.open_memmap(
                temp_dir / "y_train.npy", mode="w+", dtype=np.uint8,
                shape=(train_rows,),
            )
            position = 0
            observed = np.zeros(2, dtype=np.int64)
            for chunk in pd.read_csv(
                train_path, compression="gzip", chunksize=CHUNK_ROWS,
                dtype=p2_data.data_dtypes(context), float_precision="round_trip",
            ):
                labels = p2_data.validate_chunk(chunk, context, TRAIN_DAYS)
                stop = position + len(chunk)
                if stop > train_rows:
                    raise ValueError("Train P2 có nhiều dòng hơn split manifest.")
                values = chunk[features].to_numpy(dtype=np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise ValueError("Train P2 có feature không hữu hạn sau đổi float32.")
                x_map[position:stop] = values
                y_map[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != train_rows or observed.tolist() != [
                int(split_counts[label]) for label in LABELS
            ]:
                raise ValueError("Train P2 không khớp số dòng/nhãn manifest.")
            p2_data.check_file_signature(train_path, signature)
            fit_seconds = _fit(model, x_map, y_map)
        finally:
            if x_map is not None:
                x_map._mmap.close()
            if y_map is not None:
                y_map._mmap.close()

        temporary_model = temp_dir / model_path.name
        joblib.dump(model, temporary_model, compress=3)
        summary = {
            "protocol": "P2",
            "source_run_id": context["validation_manifest"]["source_run_id"],
            "validation_run_id": context["validation_manifest"]["validation_run_id"],
            "fit_partition": "Train",
            "train_rows": train_rows,
            "train_label_counts": split_counts,
            "feature_columns": features,
            "selected_parameters": selection["selected_parameters"],
            "fit_seconds": fit_seconds,
            "tree_depth_observed": int(model.get_depth()),
            "tree_leaf_count": int(model.get_n_leaves()),
            "test_read_or_used": False,
            "test_features_used_to_filter_train": True,
        }
        temporary_summary = temp_dir / summary_path.name
        with temporary_summary.open("x", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        temporary_model.replace(model_path)
        try:
            temporary_summary.replace(summary_path)
        except OSError:
            model_path.unlink()
            raise
    print(f"Đã lưu Decision Tree P2: {model_path}")
    return model_path


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    model_path = run_dir / "p2_decision_tree_selected_model.joblib"
    summary_path = run_dir / "p2_decision_tree_selected_training.json"
    output = run_dir / "p2_decision_tree_test_metrics.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    if not model_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Fit Decision Tree trên Train P2 trước khi đánh giá Test.")
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
        != sum(int(value) for value in context["split_manifest"]["counts"]["Train"].values())
        or training.get("test_read_or_used") is not False
        or training.get("test_features_used_to_filter_train") is not True
    ):
        raise ValueError("Model Decision Tree đã fit không khớp run/cấu hình P2.")
    model = joblib.load(model_path)
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
