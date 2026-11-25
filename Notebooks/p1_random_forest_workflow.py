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

import validation_models


CHUNK_ROWS = 20_000
LABEL_TO_CODE = {"Normal": 0, "Attack": 1}
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


def _context(validation_run_dir: Path) -> dict:
    return validation_models._load_context(
        validation_run_dir, "P1", current_p1_run=True
    )


def current_validation_run_dir(source_run_dir: Path | None = None) -> Path:
    return validation_models.latest_p1_validation_run_dir(source_run_dir)


def _parameters(candidate: dict) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown Random Forest candidate: {candidate}")
    return {**FIXED_PARAMETERS, **candidate}


def _new_model(parameters: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**parameters)


def _write_json_new(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    with path.open("x", encoding="utf-8") as file:
        try:
            file.write(text)
        except BaseException:
            file.close()
            path.unlink()
            raise


def _check_fit_memory(matrix_bytes: int) -> None:
    node_bytes = FIXED_PARAMETERS["n_estimators"] * (2 * FIXED_PARAMETERS["max_leaf_nodes"] - 1) * 128
    estimated_peak = int(matrix_bytes * 3 + node_bytes + 512 * 1024**2)
    available = psutil.virtual_memory().available
    print(
        f"Peak RAM ước tính: {estimated_peak / 1024**3:.2f} GiB; "
        f"RAM khả dụng: {available / 1024**3:.2f} GiB."
    )
    if estimated_peak > available * 0.70:
        raise MemoryError("Không đủ RAM để fit Random Forest an toàn.")


def _fit(model: RandomForestClassifier, x: np.ndarray, y: np.ndarray) -> float:
    _check_fit_memory(x.nbytes)
    started = time.perf_counter()
    model.fit(x, y)
    return time.perf_counter() - started


def _evaluate(
    model: RandomForestClassifier, path: Path, context: dict, partition: str
) -> dict:
    return validation_models.evaluate_binary_classifier(model, path, context, partition)


def prepare_validation_cache(validation_run_dir: Path) -> dict:
    context = _context(validation_run_dir)
    rows = int(context["manifest"]["partition_rows"]["TrainInternal"])
    _check_fit_memory(rows * len(context["feature_columns"]) * 4)
    return validation_models.prepare_training_cache(
        validation_run_dir, "P1", current_p1_run=True
    )


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = _context(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "random_forest_validation_selection.json"
    if output.exists():
        raise FileExistsError(f"Kết quả lựa chọn đã tồn tại: {output}")

    cache_dir = run_dir / "training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError("Chạy prepare_validation_cache trước khi so sánh.")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    rows = int(context["manifest"]["partition_rows"]["TrainInternal"])
    if (
        x.shape != (rows, len(context["feature_columns"]))
        or x.dtype != np.dtype(np.float32)
        or y.shape != (rows,)
        or y.dtype != np.dtype(np.uint8)
    ):
        raise ValueError("Cache Train nội bộ không khớp manifest.")
    expected_counts = context["manifest"]["partition_label_counts"][
        "TrainInternal"
    ]
    if np.bincount(y, minlength=2).tolist() != [
        int(expected_counts[label]) for label in LABEL_TO_CODE
    ]:
        raise ValueError("Nhãn trong cache không khớp audit Train nội bộ.")

    results = []
    try:
        for candidate in CANDIDATES:
            parameters = _parameters(candidate)
            print(
                "Đang fit và đánh giá Validation: "
                f"max_depth={candidate['max_depth']}, "
                f"min_samples_leaf={candidate['min_samples_leaf']}"
            )
            model = _new_model(parameters)
            fit_seconds = _fit(model, x, y)
            metrics = _evaluate(
                model, context["validation_path"], context, "Validation"
            )
            results.append(
                {
                    "max_depth": candidate["max_depth"],
                    "min_samples_leaf": candidate["min_samples_leaf"],
                    "max_leaf_nodes": FIXED_PARAMETERS["max_leaf_nodes"],
                    "fit_seconds": fit_seconds,
                    "max_tree_depth_observed": int(max(tree.get_depth() for tree in model.estimators_)),
                    "total_leaf_count": int(sum(tree.get_n_leaves() for tree in model.estimators_)),
                    **metrics,
                }
            )
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
            row["min_samples_leaf"],
        ),
    )
    selected_parameters = _parameters(
        {
            "max_depth": selected["max_depth"],
            "min_samples_leaf": selected["min_samples_leaf"],
        }
    )
    summary = {
        "source_run_id": context["manifest"]["source_run_id"],
        "validation_run_id": context["manifest"]["validation_run_id"],
        "selected_parameters": selected_parameters,
        "selection_rule": (
            "highest macro_f1; on exact tie, lower max_depth then "
            "higher min_samples_leaf"
        ),
        "test_read_or_used": False,
        "results": results,
    }
    _write_json_new(output, summary)
    print(f"Cấu hình được chọn theo Macro-F1: {selected_parameters}")
    return pd.DataFrame(results)


def _selected_context(validation_run_dir: Path) -> tuple[dict, dict]:
    context = _context(validation_run_dir)
    selection_path = (
        context["run_dir"] / "random_forest_validation_selection.json"
    )
    with selection_path.open(encoding="utf-8") as file:
        selection = json.load(file)
    manifest = context["manifest"]
    parameters = selection.get("selected_parameters")
    if (
        selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("test_read_or_used") is not False
        or parameters
        not in [
            _parameters(candidate)
            for candidate in CANDIDATES
        ]
    ):
        raise ValueError("Lựa chọn Random Forest không khớp run Validation.")
    return context, selection


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _selected_context(validation_run_dir)
    manifest = context["manifest"]
    source_dir = context["run_dir"].resolve().parent.parent
    train_path = source_dir / "train.csv.gz"
    output = context["run_dir"] / "random_forest_selected_model.joblib"
    summary_path = context["run_dir"] / "random_forest_selected_training.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary đã tồn tại; không ghi đè.")
    if not train_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy Train gốc: {train_path}")

    split_manifest_path = source_dir / "split_manifest.json"
    with split_manifest_path.open(encoding="utf-8") as file:
        split_manifest = json.load(file)
    if (
        split_manifest.get("run_id") != manifest["source_run_id"]
        or split_manifest.get("feature_columns") != context["feature_columns"]
        or split_manifest.get("split_csv_float_precision") != "round_trip"
    ):
        raise ValueError("Manifest Train gốc không khớp Validation P1.")
    train_counts = split_manifest["split_counts"]["Train"]
    rows = int(train_counts["rows"])
    if rows != int(manifest["source_train_rows"]):
        raise ValueError("Số dòng Train trong hai manifest không khớp.")
    counts = manifest["source_train_label_counts"]
    if any(
        int(train_counts[label]) != int(counts[label])
        for label in LABEL_TO_CODE
    ):
        raise ValueError("Phân bố nhãn Train trong hai manifest không khớp.")
    matrix_bytes = (
        rows
        * len(context["feature_columns"])
        * np.dtype(np.float32).itemsize
    )
    _check_fit_memory(matrix_bytes)
    model_budget = FIXED_PARAMETERS["n_estimators"] * (2 * FIXED_PARAMETERS["max_leaf_nodes"] - 1) * 128
    if (
        shutil.disk_usage(context["run_dir"]).free
        < matrix_bytes + rows + model_budget + 64 * 1024**2
    ):
        raise OSError("Không đủ đĩa cho cache Train gốc.")

    model = _new_model(selection["selected_parameters"])
    with tempfile.TemporaryDirectory(
        prefix=".random_forest_fit_", dir=context["run_dir"]
    ) as temp:
        x = None
        y = None
        try:
            x = np.lib.format.open_memmap(
                Path(temp) / "x.npy",
                mode="w+",
                dtype=np.float32,
                shape=(rows, len(context["feature_columns"])),
            )
            y = np.lib.format.open_memmap(
                Path(temp) / "y.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(rows,),
            )
            observed = np.zeros(2, dtype=np.int64)
            position = 0
            for chunk in pd.read_csv(
                train_path,
                compression="gzip",
                chunksize=CHUNK_ROWS,
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
                    raise ValueError(
                        "Chuyển feature sang float32 tạo giá trị không hữu hạn."
                    )
                x[position:stop] = values
                y[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != rows or observed.tolist() != [
                int(counts[label]) for label in LABEL_TO_CODE
            ]:
                raise ValueError("Số dòng hoặc nhãn Train gốc không khớp audit.")
            fit_seconds = _fit(model, x, y)
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
            "fit_seconds": fit_seconds,
            "n_estimators_observed": len(model.estimators_),
            "max_tree_depth_observed": int(max(tree.get_depth() for tree in model.estimators_)),
            "total_leaf_count": int(sum(tree.get_n_leaves() for tree in model.estimators_)),
            "test_read_or_used": False,
        }
        temporary_summary = Path(temp) / "training.json"
        with temporary_summary.open("x", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        with temporary_model.open("rb") as source, output.open("xb") as target:
            try:
                shutil.copyfileobj(source, target)
            except BaseException:
                target.close()
                output.unlink()
                raise
        try:
            with temporary_summary.open("rb") as source, summary_path.open("xb") as target:
                try:
                    shutil.copyfileobj(source, target)
                except BaseException:
                    target.close()
                    summary_path.unlink()
                    raise
        except BaseException:
            output.unlink()
            raise

    print(f"Đã lưu Random Forest: {output}")
    return output


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _selected_context(validation_run_dir)
    output = context["run_dir"] / "random_forest_test_metrics.json"
    model_path = context["run_dir"] / "random_forest_selected_model.joblib"
    summary_path = context["run_dir"] / "random_forest_selected_training.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    if not model_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Fit Random Forest trên Train trước khi đánh giá Test.")
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    if (
        summary.get("source_run_id") != context["manifest"]["source_run_id"]
        or summary.get("validation_run_id") != context["manifest"]["validation_run_id"]
        or summary.get("selected_parameters") != selection["selected_parameters"]
        or summary.get("feature_columns") != context["feature_columns"]
        or summary.get("fit_partition") != "Train"
        or summary.get("train_rows") != context["manifest"]["source_train_rows"]
        or summary.get("test_read_or_used") is not False
    ):
        raise ValueError("Model đã fit không khớp cấu hình hoặc run đã chọn.")

    source_dir = context["run_dir"].resolve().parent.parent
    with (source_dir / "split_manifest.json").open(encoding="utf-8") as file:
        split_manifest = json.load(file)
    if (
        split_manifest.get("run_id") != context["manifest"]["source_run_id"]
        or split_manifest.get("feature_columns") != context["feature_columns"]
    ):
        raise ValueError("Manifest Test không khớp run nguồn.")
    counts = split_manifest["split_counts"]["Test"]
    test_path = source_dir / "test.csv.gz"
    if not test_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy Test P1: {test_path}")
    test_context = {
        **context,
        "manifest": {
            **context["manifest"],
            "partition_rows": {"Test": int(counts["rows"])},
            "partition_label_counts": {
                "Test": {
                    label: int(counts[label])
                    for label in LABEL_TO_CODE
                }
            },
        },
    }
    model = joblib.load(model_path)
    if list(model.classes_) != [0, 1]:
        raise ValueError("Random Forest classes không khớp Normal=0/Attack=1.")
    metrics = _evaluate(model, test_path, test_context, "Test")
    _write_json_new(
        output,
        {
            "source_run_id": summary["source_run_id"],
            "validation_run_id": summary["validation_run_id"],
            "selected_parameters": summary["selected_parameters"],
            "partition": "Test",
            "test_read_or_used": True,
            "metrics": metrics,
        },
    )
    return pd.DataFrame([metrics])
