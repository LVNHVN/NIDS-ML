from __future__ import annotations

import gc
import json
import os
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
LABELS = ("Normal", "Attack")
LABEL_TO_CODE = {"Normal": 0, "Attack": 1}
EXPECTED_METADATA = ["SourceFile", "CaptureDay", "OriginalLabel", "BinaryLabel"]
TRAIN_DAYS = {"Monday", "Tuesday", "Wednesday"}
TEST_DAYS = {"Thursday", "Friday"}
SPLIT_POLICY = "keep_all_test_and_remove_train_rows_whose_feature_hash_occurs_in_test"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def _signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _check_signature(path: Path, signature: dict) -> None:
    if _signature(path) != signature:
        raise RuntimeError(f"Artifact đã thay đổi trong quá trình xử lý: {path}")


def _load_context(validation_run_dir: Path) -> dict:
    run_dir = Path(validation_run_dir).resolve()
    validation_manifest_path = run_dir / "validation_manifest.json"
    if not validation_manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy Validation manifest: {validation_manifest_path}")
    validation_manifest = _read_json(validation_manifest_path)
    if run_dir.parent.name != "validation_runs":
        raise ValueError("Validation run không nằm trong validation_runs.")
    source_dir = run_dir.parent.parent
    split_manifest_path = source_dir / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy split manifest: {split_manifest_path}")
    split_manifest = _read_json(split_manifest_path)
    project_root = Path(__file__).resolve().parents[1]
    expected_root = (project_root / "Dataset" / "Processed" / "P2_Day_Disjoint").resolve()
    if (
        source_dir.parent.resolve() != expected_root
        or validation_manifest.get("validation_run_id") != run_dir.name
        or validation_manifest.get("source_run_id") != source_dir.name
        or validation_manifest.get("protocol") != "P2"
        or validation_manifest.get("validation_method") != "GroupShuffleSplit"
        or validation_manifest.get("test_read_during_validation") is not False
        or validation_manifest.get("test_features_used_to_filter_source_train") is not True
        or validation_manifest.get("shared_feature_groups") != 0
        or validation_manifest.get("model_fitted") is not False
        or validation_manifest.get("split_csv_float_precision") != "round_trip"
        or split_manifest.get("run_id") != source_dir.name
        or split_manifest.get("split_method") != "Day-Disjoint"
        or split_manifest.get("overlap_policy") != SPLIT_POLICY
        or split_manifest.get("test_features_used_to_filter_train") is not True
        or split_manifest.get("shared_feature_groups_after_filtering") != 0
        or split_manifest.get("training_started") is not False
        or split_manifest.get("split_csv_float_precision") != "round_trip"
    ):
        raise ValueError("Manifest Validation hoặc split không khớp protocol P2 đã duyệt.")

    feature_columns = validation_manifest.get("feature_columns", [])
    metadata_columns = validation_manifest.get("metadata_columns", [])
    columns = feature_columns + metadata_columns
    if len(feature_columns) != 78 or len(set(feature_columns)) != 78:
        raise ValueError("P2 Logistic cần đúng 78 feature duy nhất.")
    if metadata_columns != EXPECTED_METADATA or len(columns) != 82:
        raise ValueError("Schema metadata P2 không khớp.")

    partition_rows = validation_manifest.get("partition_rows", {})
    partition_counts = validation_manifest.get("partition_label_counts", {})
    source_train_counts = split_manifest.get("counts", {}).get("Train", {})
    if set(partition_rows) != {"TrainInternal", "Validation"}:
        raise ValueError("Validation manifest thiếu partition.")
    for label in LABELS:
        if sum(
            int(partition_counts.get(part, {}).get(label, -1))
            for part in ("TrainInternal", "Validation")
        ) != int(source_train_counts.get(label, -2)):
            raise ValueError("Train nội bộ và Validation không bảo toàn nhãn Train P2.")
    for part in ("TrainInternal", "Validation"):
        counts = partition_counts.get(part, {})
        if (
            int(partition_rows[part]) <= 0
            or sum(int(counts.get(label, 0)) for label in LABELS)
            != int(partition_rows[part])
            or any(int(counts.get(label, 0)) <= 0 for label in LABELS)
        ):
            raise ValueError(f"Số dòng/nhãn không hợp lệ ở {part}.")
    if sum(int(partition_rows[part]) for part in partition_rows) != sum(
        int(source_train_counts[label]) for label in LABELS
    ):
        raise ValueError("Validation không bảo toàn tổng số dòng Train P2.")

    paths = {
        "train_internal": run_dir / "train_internal.csv.gz",
        "validation": run_dir / "validation.csv.gz",
        "train": source_dir / "p2_train.csv.gz",
        "test": source_dir / "p2_test.csv.gz",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy artifact {name}: {path}")
        header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
        if header != columns:
            raise ValueError(f"Schema {name} không khớp manifest.")

    test_counts = split_manifest.get("counts", {}).get("Test", {})
    if any(int(test_counts.get(label, 0)) <= 0 for label in LABELS):
        raise ValueError("Test P2 phải có cả Normal và Attack.")
    return {
        "run_dir": run_dir,
        "source_dir": source_dir,
        "validation_manifest": validation_manifest,
        "split_manifest": split_manifest,
        "feature_columns": feature_columns,
        "metadata_columns": metadata_columns,
        "columns": columns,
        "paths": paths,
    }


def latest_validation_run_dir(
    required_artifacts: tuple[str, ...] = (),
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    p2_root = project_root / "Dataset" / "Processed" / "P2_Day_Disjoint"
    if not p2_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục P2: {p2_root}")
    candidates = []
    for manifest_path in p2_root.glob("*/validation_runs/*/validation_manifest.json"):
        run_dir = manifest_path.parent
        if not all((run_dir / name).is_file() for name in required_artifacts):
            continue
        manifest = _read_json(manifest_path)
        if (
            manifest.get("protocol") == "P2"
            and manifest.get("source_run_id") == run_dir.parent.parent.name
            and manifest.get("validation_run_id") == run_dir.name
        ):
            candidates.append((run_dir.parent.parent.name, run_dir.name, run_dir))
    if not candidates:
        raise FileNotFoundError("Không tìm thấy Validation P2 phù hợp với artifact yêu cầu.")
    _, _, run_dir = max(candidates, key=lambda item: (item[0], item[1]))
    _load_context(run_dir)
    return run_dir


def _dtypes(context: dict) -> dict:
    return {
        **{column: np.float64 for column in context["feature_columns"]},
        **{column: str for column in context["metadata_columns"]},
    }


def _validated(chunk: pd.DataFrame, context: dict, expected_days: set[str]) -> np.ndarray:
    labels = validation_models._validate_chunk(
        chunk,
        {
            "columns": context["columns"],
            "feature_columns": context["feature_columns"],
            "metadata_columns": context["metadata_columns"],
        },
    )
    if not chunk["CaptureDay"].isin(expected_days).all():
        raise ValueError("CaptureDay nằm ngoài partition P2 dự kiến.")
    return labels


def _cache_identity(context: dict, signature: dict) -> dict:
    manifest = context["validation_manifest"]
    return {
        "source_run_id": manifest["source_run_id"],
        "validation_run_id": manifest["validation_run_id"],
        "source_file": "train_internal.csv.gz",
        "source_signature": signature,
        "feature_columns": context["feature_columns"],
        "partition_rows": manifest["partition_rows"]["TrainInternal"],
        "partition_label_counts": manifest["partition_label_counts"]["TrainInternal"],
        "dtype": "float32",
    }


def prepare_validation_cache(validation_run_dir: Path) -> dict:
    context = _load_context(validation_run_dir)
    run_dir = context["run_dir"]
    path = context["paths"]["train_internal"]
    signature = _signature(path)
    identity = _cache_identity(context, signature)
    cache_dir = run_dir / "p2_logistic_training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"
    cache_manifest_path = cache_dir / "cache_manifest.json"
    rows = int(identity["partition_rows"])
    features = context["feature_columns"]
    expected_counts = identity["partition_label_counts"]

    if cache_dir.exists():
        if not x_path.is_file() or not y_path.is_file() or not cache_manifest_path.is_file():
            raise FileExistsError(f"Cache chưa hoàn chỉnh, không ghi đè: {cache_dir}")
        if _read_json(cache_manifest_path) != identity:
            raise ValueError("Cache đã có nhưng không khớp Validation hiện tại.")
        x_train = np.load(x_path, mmap_mode="r")
        y_train = np.load(y_path, mmap_mode="r")
        if (
            x_train.shape != (rows, len(features))
            or x_train.dtype != np.float32
            or y_train.shape != (rows,)
            or y_train.dtype != np.uint8
            or np.bincount(y_train, minlength=2).tolist()
            != [int(expected_counts[label]) for label in LABELS]
        ):
            raise ValueError("Cache shape, dtype hoặc nhãn không khớp manifest.")
        del x_train, y_train
        gc.collect()
        _check_signature(path, signature)
        print(f"Đã dùng cache đã xác minh: {cache_dir}")
        return context

    matrix_bytes = rows * len(features) * np.dtype(np.float32).itemsize
    required_disk = matrix_bytes + rows + 64 * 1024**2
    if shutil.disk_usage(run_dir).free < required_disk:
        raise OSError("Không đủ dung lượng đĩa cho cache Train nội bộ.")
    working_bytes = CHUNK_ROWS * len(context["columns"]) * 8 * 2 + 2 * 1024**2
    available = psutil.virtual_memory().available
    if working_bytes > available * 0.5:
        raise MemoryError("Không đủ RAM cho tạo cache theo chunk.")

    temp_dir = Path(tempfile.mkdtemp(prefix=".p2_logistic_cache_", dir=run_dir))
    x_map = y_map = None
    try:
        x_temp = temp_dir / x_path.name
        y_temp = temp_dir / y_path.name
        x_map = np.lib.format.open_memmap(
            x_temp, mode="w+", dtype=np.float32, shape=(rows, len(features))
        )
        y_map = np.lib.format.open_memmap(
            y_temp, mode="w+", dtype=np.uint8, shape=(rows,)
        )
        position = 0
        observed = np.zeros(2, dtype=np.int64)
        for chunk in pd.read_csv(
            path, compression="gzip", chunksize=CHUNK_ROWS,
            dtype=_dtypes(context), float_precision="round_trip",
        ):
            labels = _validated(chunk, context, TRAIN_DAYS)
            stop = position + len(chunk)
            if stop > rows:
                raise ValueError("Train nội bộ có nhiều dòng hơn manifest.")
            values = chunk[features].to_numpy(dtype=np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError("Chuyển feature sang float32 tạo giá trị không hữu hạn.")
            x_map[position:stop] = values
            y_map[position:stop] = labels
            observed += np.bincount(labels, minlength=2)
            position = stop
        if position != rows or observed.tolist() != [
            int(expected_counts[label]) for label in LABELS
        ]:
            raise ValueError("Train nội bộ không khớp số dòng/nhãn trong manifest.")
        _check_signature(path, signature)
        x_map.flush()
        y_map.flush()
        del x_map, y_map
        x_map = y_map = None
        _write_json(temp_dir / cache_manifest_path.name, identity)
        os.replace(temp_dir, cache_dir)
    finally:
        if x_map is not None:
            del x_map
        if y_map is not None:
            del y_map
        gc.collect()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    print(f"Cache TrainInternal: {rows:,} dòng × {len(features)} feature.")
    return context


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    solver="saga", C=c_value, max_iter=200, tol=1e-3,
                    class_weight=None, random_state=42,
                ),
            ),
        ]
    )


def _fit(pipeline: Pipeline, x_train: np.ndarray, y_train: np.ndarray) -> float:
    estimated_peak = int(x_train.nbytes * 4)
    available = psutil.virtual_memory().available
    print(
        f"Peak RAM ước tính: {estimated_peak / 1024**3:.2f} GiB; "
        f"RAM khả dụng: {available / 1024**3:.2f} GiB."
    )
    if estimated_peak > available * 0.70:
        raise MemoryError("Ước tính bộ nhớ vượt ngưỡng an toàn cho Logistic.")
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(x_train, y_train)
    return time.perf_counter() - started


def _evaluate(
    model: validation_models.BinaryPredictor,
    path: Path,
    context: dict,
    partition: str,
    expected_rows: int,
    expected_counts: dict,
    expected_days: set[str],
) -> dict:
    confusion = np.zeros((2, 2), dtype=np.int64)
    observed = np.zeros(2, dtype=np.int64)
    signature = _signature(path)
    for chunk in pd.read_csv(
        path, compression="gzip", chunksize=CHUNK_ROWS,
        dtype=_dtypes(context), float_precision="round_trip",
    ):
        labels = _validated(chunk, context, expected_days)
        values = chunk[context["feature_columns"]].to_numpy(
            dtype=np.float32, copy=False
        )
        if not np.isfinite(values).all():
            raise ValueError("Feature dự đoán có giá trị không hữu hạn.")
        predictions = model.predict(values)
        confusion += confusion_matrix(labels, predictions, labels=[0, 1])
        observed += np.bincount(labels, minlength=2)
    _check_signature(path, signature)
    expected = [int(expected_counts[label]) for label in LABELS]
    if observed.tolist() != expected or int(observed.sum()) != expected_rows:
        raise ValueError(f"Số dòng/nhãn {partition} không khớp manifest.")
    tn, fp, fn, tp = (int(value) for value in confusion.ravel())
    if tn + fp == 0 or tp + fn == 0:
        raise ValueError(f"{partition} phải có cả Normal và Attack.")
    attack_precision = tp / (tp + fp) if tp + fp else 0.0
    attack_recall = tp / (tp + fn)
    attack_f1 = 2 * tp / (2 * tp + fp + fn)
    normal_f1 = 2 * tn / (2 * tn + fp + fn)
    return {
        "rows": expected_rows, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": (tn + tp) / expected_rows,
        "attack_precision": attack_precision, "attack_recall": attack_recall,
        "attack_f1": attack_f1, "macro_f1": (normal_f1 + attack_f1) / 2,
        "fpr": fp / (tn + fp), "fnr": fn / (tp + fn),
    }


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = prepare_validation_cache(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "p2_logistic_validation_selection.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Validation đã tồn tại: {output}")
    cache_dir = run_dir / "p2_logistic_training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError("Hãy chạy prepare_validation_cache trước.")
    rows = int(context["validation_manifest"]["partition_rows"]["TrainInternal"])
    x_train = np.load(x_path, mmap_mode="r")
    y_train = np.load(y_path, mmap_mode="r")
    if (
        x_train.shape != (rows, len(context["feature_columns"]))
        or y_train.shape != (rows,)
    ):
        raise ValueError("Cache TrainInternal không khớp manifest.")
    results = []
    try:
        for c_value in C_VALUES:
            print(f"Đang so sánh Logistic trên Validation: C={c_value}")
            pipeline = _pipeline(c_value)
            fit_seconds = _fit(pipeline, x_train, y_train)
            metrics = _evaluate(
                pipeline, context["paths"]["validation"], context, "Validation",
                int(context["validation_manifest"]["partition_rows"]["Validation"]),
                context["validation_manifest"]["partition_label_counts"]["Validation"],
                TRAIN_DAYS,
            )
            results.append({
                "C": c_value, "fit_seconds": fit_seconds,
                "n_iter": int(pipeline.named_steps["logistic_regression"].n_iter_[0]),
                **metrics,
            })
            del pipeline
            gc.collect()
    finally:
        del x_train, y_train
        gc.collect()
    selected = max(results, key=lambda row: (row["macro_f1"], -row["C"]))
    summary = {
        "protocol": "P2",
        "source_run_id": context["validation_manifest"]["source_run_id"],
        "validation_run_id": context["validation_manifest"]["validation_run_id"],
        "selected_C": selected["C"],
        "model_parameters": {
            "solver": "saga",
            "max_iter": 200,
            "tol": 1e-3,
            "class_weight": None,
            "random_state": 42,
        },
        "selection_rule": "highest macro_f1; smallest C on exact tie",
        "test_read_or_used": False,
        "test_features_used_to_filter_train": True,
        "results": results,
    }
    _write_json(output, summary)
    print(f"C được chọn theo Macro-F1 Validation: {selected['C']}")
    return pd.DataFrame(results)


def _selected_context(validation_run_dir: Path) -> tuple[dict, dict]:
    context = _load_context(validation_run_dir)
    selection = _read_json(context["run_dir"] / "p2_logistic_validation_selection.json")
    manifest = context["validation_manifest"]
    if (
        selection.get("protocol") != "P2"
        or selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("test_read_or_used") is not False
        or selection.get("test_features_used_to_filter_train") is not True
        or selection.get("selected_C") not in C_VALUES
    ):
        raise ValueError("Lựa chọn Logistic không khớp Validation P2.")
    return context, selection


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _selected_context(validation_run_dir)
    run_dir = context["run_dir"]
    train_path = context["paths"]["train"]
    model_path = run_dir / "p2_logistic_selected_pipeline.joblib"
    summary_path = run_dir / "p2_logistic_selected_training.json"
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary P2 Logistic đã tồn tại; không ghi đè.")

    split_counts = context["split_manifest"]["counts"]["Train"]
    rows = sum(int(split_counts[label]) for label in LABELS)
    features = context["feature_columns"]
    matrix_bytes = rows * len(features) * np.dtype(np.float32).itemsize
    if shutil.disk_usage(run_dir).free < matrix_bytes + rows + 64 * 1024**2:
        raise OSError("Không đủ dung lượng đĩa cho fit Logistic trên Train P2.")
    signature = _signature(train_path)
    with tempfile.TemporaryDirectory(prefix=".p2_logistic_fit_", dir=run_dir) as temp:
        temp_dir = Path(temp)
        x_map = y_map = None
        try:
            x_map = np.lib.format.open_memmap(
                temp_dir / "x_train.npy", mode="w+", dtype=np.float32,
                shape=(rows, len(features)),
            )
            y_map = np.lib.format.open_memmap(
                temp_dir / "y_train.npy", mode="w+", dtype=np.uint8, shape=(rows,),
            )
            position = 0
            observed = np.zeros(2, dtype=np.int64)
            for chunk in pd.read_csv(
                train_path, compression="gzip", chunksize=CHUNK_ROWS,
                dtype=_dtypes(context), float_precision="round_trip",
            ):
                labels = _validated(chunk, context, TRAIN_DAYS)
                stop = position + len(chunk)
                if stop > rows:
                    raise ValueError("Train P2 có nhiều dòng hơn split manifest.")
                values = chunk[features].to_numpy(dtype=np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise ValueError("Train P2 có feature không hữu hạn sau đổi float32.")
                x_map[position:stop] = values
                y_map[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != rows or observed.tolist() != [
                int(split_counts[label]) for label in LABELS
            ]:
                raise ValueError("Train P2 không khớp số dòng/nhãn trong manifest.")
            _check_signature(train_path, signature)
            pipeline = _pipeline(float(selection["selected_C"]))
            fit_seconds = _fit(pipeline, x_map, y_map)
        finally:
            if x_map is not None:
                x_map._mmap.close()
            if y_map is not None:
                y_map._mmap.close()
        temporary_model = temp_dir / model_path.name
        joblib.dump(pipeline, temporary_model, compress=3)
        summary = {
            "protocol": "P2",
            "source_run_id": context["validation_manifest"]["source_run_id"],
            "validation_run_id": context["validation_manifest"]["validation_run_id"],
            "selected_C": selection["selected_C"],
            "model_parameters": {
                "solver": "saga",
                "max_iter": 200,
                "tol": 1e-3,
                "class_weight": None,
                "random_state": 42,
            },
            "feature_columns": features,
            "train_rows": rows,
            "train_label_counts": split_counts,
            "fit_seconds": fit_seconds,
            "n_iter": int(pipeline.named_steps["logistic_regression"].n_iter_[0]),
            "test_read_or_used": False,
            "test_features_used_to_filter_train": True,
        }
        temporary_summary = temp_dir / summary_path.name
        _write_json(temporary_summary, summary)
        temporary_model.replace(model_path)
        try:
            temporary_summary.replace(summary_path)
        except OSError:
            model_path.unlink()
            raise
    print(f"Đã lưu pipeline Logistic P2: {model_path}")
    return model_path


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _selected_context(validation_run_dir)
    run_dir = context["run_dir"]
    model_path = run_dir / "p2_logistic_selected_pipeline.joblib"
    summary_path = run_dir / "p2_logistic_selected_training.json"
    output = run_dir / "p2_logistic_test_metrics.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    summary = _read_json(summary_path)
    manifest = context["validation_manifest"]
    if (
        summary.get("protocol") != "P2"
        or summary.get("source_run_id") != manifest["source_run_id"]
        or summary.get("validation_run_id") != manifest["validation_run_id"]
        or summary.get("selected_C") != selection["selected_C"]
        or summary.get("feature_columns") != context["feature_columns"]
        or summary.get("test_read_or_used") is not False
        or summary.get("test_features_used_to_filter_train") is not True
        or not model_path.is_file()
    ):
        raise ValueError("Pipeline Logistic P2 không khớp lựa chọn/Train.")
    model = joblib.load(model_path)
    split_counts = context["split_manifest"]["counts"]["Test"]
    test_rows = sum(int(split_counts[label]) for label in LABELS)
    metrics = _evaluate(
        model, context["paths"]["test"], context, "Test", test_rows,
        split_counts, TEST_DAYS,
    )
    result = {
        "protocol": "P2",
        "source_run_id": manifest["source_run_id"],
        "validation_run_id": manifest["validation_run_id"],
        "selected_C": selection["selected_C"],
        "partition": "Test",
        "previously_seen_test": True,
        "test_read_or_used": True,
        "test_features_used_to_filter_train": True,
        "metrics": metrics,
    }
    _write_json(output, result)
    return pd.DataFrame([metrics])


def evaluate_partition(
    model: validation_models.BinaryPredictor,
    validation_run_dir: Path,
    partition: str,
) -> dict:
    context = _load_context(validation_run_dir)
    if partition == "Validation":
        manifest = context["validation_manifest"]
        rows = int(manifest["partition_rows"]["Validation"])
        counts = manifest["partition_label_counts"]["Validation"]
        path = context["paths"]["validation"]
        expected_days = TRAIN_DAYS
    elif partition == "Test":
        counts = context["split_manifest"]["counts"]["Test"]
        rows = sum(int(counts[label]) for label in LABELS)
        path = context["paths"]["test"]
        expected_days = TEST_DAYS
    else:
        raise ValueError(f"Unsupported P2 evaluation partition: {partition}")
    return _evaluate(
        model, path, context, partition, rows, counts, expected_days
    )


load_context = _load_context
data_dtypes = _dtypes
validate_chunk = _validated
file_signature = _signature
check_file_signature = _check_signature
