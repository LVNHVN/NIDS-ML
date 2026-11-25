from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil
import tempfile
import time
import warnings

import joblib
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import psutil
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import p2_logistic_workflow as p2_data


CHUNK_ROWS = 20_000
LABELS = ("Normal", "Attack")
TRAIN_DAYS = {"Monday", "Tuesday", "Wednesday"}
TEST_DAYS = {"Thursday", "Friday"}
CANDIDATE_HIDDEN_UNITS = (32, 64, 128)
FIXED_PARAMETERS = {
    "activation": "relu",
    "alpha": 0.0001,
    "batch_size": 1024,
    "early_stopping": False,
    "learning_rate_init": 0.001,
    "max_iter": 200,
    "random_state": 42,
    "solver": "adam",
}
MIN_AVAILABLE_RAM = 4 * 1024**3
MIN_FREE_DISK = 3 * 1024**3


def latest_validation_run_dir(
    required_artifacts: tuple[str, ...] = (),
) -> Path:
    return p2_data.latest_validation_run_dir(
        required_artifacts=required_artifacts
    )


def _parameters(hidden_units: int) -> dict:
    if hidden_units not in CANDIDATE_HIDDEN_UNITS:
        raise ValueError(f"Cấu hình MLP P2 không hợp lệ: {hidden_units}")
    return {
        **FIXED_PARAMETERS,
        "hidden_layer_sizes": [hidden_units],
    }


def _pipeline(hidden_units: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    **FIXED_PARAMETERS,
                    hidden_layer_sizes=(hidden_units,),
                ),
            ),
        ]
    )


def _check_resources(run_dir: Path, matrix_bytes: int) -> None:
    available = psutil.virtual_memory().available
    free = shutil.disk_usage(run_dir).free
    estimated_peak = int(matrix_bytes * 4 + 128 * 1024**2)
    print(
        f"Peak RAM ước tính: {estimated_peak / 1024**3:.2f} GiB; "
        f"RAM khả dụng: {available / 1024**3:.2f} GiB; "
        f"đĩa trống: {free / 1024**3:.2f} GiB. "
        "Ước tính không bảo đảm peak RAM thực tế."
    )
    if available < MIN_AVAILABLE_RAM:
        raise MemoryError("Cần ít nhất 4 GiB RAM khả dụng trước khi fit MLP P2.")
    if estimated_peak > available * 0.70:
        raise MemoryError(
            "Peak RAM ước tính vượt 70% RAM khả dụng; dừng trước khi fit MLP P2."
        )
    if free < MIN_FREE_DISK:
        raise OSError("Cần ít nhất 3 GiB đĩa trống trước khi fit MLP P2.")


def _fit(
    pipeline: Pipeline, x: np.ndarray, y: np.ndarray, run_dir: Path
) -> tuple[float, int, float, list[str]]:
    _check_resources(run_dir, int(x.nbytes))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        started = time.perf_counter()
        pipeline.fit(x, y)
        fit_seconds = time.perf_counter() - started

    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    for message in convergence_warnings:
        print(f"ConvergenceWarning: {message}")

    estimator = pipeline.named_steps["mlp"]
    return (
        fit_seconds,
        int(estimator.n_iter_),
        float(estimator.loss_),
        convergence_warnings,
    )


def _write_json_new(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as file:
        try:
            file.write(text)
        except BaseException:
            file.close()
            path.unlink()
            raise


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    context = p2_data.load_context(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / "p2_mlp_validation_selection.json"
    if output.exists():
        raise FileExistsError(
            f"Kết quả Validation MLP đã tồn tại; không ghi đè: {output}"
        )

    context = p2_data.prepare_validation_cache(validation_run_dir)
    cache_dir = run_dir / "p2_logistic_training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError("Không tìm thấy cache TrainInternal P2.")

    manifest = context["validation_manifest"]
    rows = int(manifest["partition_rows"]["TrainInternal"])
    counts = manifest["partition_label_counts"]["TrainInternal"]
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    try:
        if (
            x.shape != (rows, len(context["feature_columns"]))
            or x.dtype != np.dtype(np.float32)
            or y.shape != (rows,)
            or y.dtype != np.dtype(np.uint8)
            or np.bincount(y, minlength=2).tolist()
            != [int(counts[label]) for label in LABELS]
        ):
            raise ValueError("Cache TrainInternal không khớp manifest P2.")

        results = []
        for hidden_units in CANDIDATE_HIDDEN_UNITS:
            print(
                f"Đang fit MLP P2 trên TrainInternal: "
                f"hidden_layer_sizes=({hidden_units},)"
            )
            pipeline = _pipeline(hidden_units)
            fit_seconds, n_iter, final_loss, convergence_warnings = _fit(
                pipeline, x, y, run_dir
            )
            metrics = p2_data.evaluate_partition(
                pipeline, run_dir, "Validation"
            )
            results.append(
                {
                    "hidden_units": hidden_units,
                    "fit_seconds": fit_seconds,
                    "n_iter": n_iter,
                    "final_loss": final_loss,
                    "convergence_warnings": convergence_warnings,
                    **metrics,
                }
            )
            del pipeline
            gc.collect()
    finally:
        x._mmap.close()
        y._mmap.close()
        del x, y
        gc.collect()

    selected = max(
        results, key=lambda row: (row["macro_f1"], -row["hidden_units"])
    )
    selected_parameters = _parameters(selected["hidden_units"])
    _write_json_new(
        output,
        {
            "protocol": "P2",
            "source_run_id": manifest["source_run_id"],
            "validation_run_id": manifest["validation_run_id"],
            "sklearn_version": sklearn_version,
            "selected_parameters": selected_parameters,
            "selection_rule": (
                "highest macro_f1; on exact tie, fewer hidden units"
            ),
            "early_stopping": False,
            "test_read_or_used": False,
            "test_features_used_to_filter_train": True,
            "results": results,
        },
    )
    print(f"MLP P2 được chọn theo Macro-F1: {selected_parameters}")
    if any(row["convergence_warnings"] for row in results):
        print(
            "Có cấu hình chưa hội tụ; xem cảnh báo và bảng Validation "
            "trước khi duyệt bước fit toàn Train."
        )
    return pd.DataFrame(results)


def _read_selection(validation_run_dir: Path) -> tuple[dict, dict]:
    context = p2_data.load_context(validation_run_dir)
    run_dir = context["run_dir"]
    selection_path = run_dir / "p2_mlp_validation_selection.json"
    with selection_path.open(encoding="utf-8") as file:
        selection = json.load(file)
    manifest = context["validation_manifest"]
    if (
        selection.get("protocol") != "P2"
        or selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("sklearn_version") != sklearn_version
        or selection.get("early_stopping") is not False
        or selection.get("test_read_or_used") is not False
        or selection.get("test_features_used_to_filter_train") is not True
        or selection.get("selected_parameters")
        not in [_parameters(size) for size in CANDIDATE_HIDDEN_UNITS]
    ):
        raise ValueError("Lựa chọn MLP không khớp run hoặc môi trường P2.")
    results = selection.get("results")
    if (
        not isinstance(results, list)
        or len(results) != len(CANDIDATE_HIDDEN_UNITS)
        or {row.get("hidden_units") for row in results}
        != set(CANDIDATE_HIDDEN_UNITS)
    ):
        raise ValueError("Kết quả Validation MLP thiếu cấu hình ứng viên.")
    return context, selection


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    train_path = context["paths"]["train"]
    model_path = run_dir / "p2_mlp_selected_pipeline.joblib"
    summary_path = run_dir / "p2_mlp_selected_training.json"
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary MLP P2 đã tồn tại; không ghi đè.")

    split_counts = context["split_manifest"]["counts"]["Train"]
    rows = sum(int(split_counts[label]) for label in LABELS)
    features = context["feature_columns"]
    matrix_bytes = rows * len(features) * np.dtype(np.float32).itemsize
    if shutil.disk_usage(run_dir).free < matrix_bytes + rows + 64 * 1024**2:
        raise OSError("Không đủ dung lượng đĩa cho fit MLP trên Train P2.")
    signature = p2_data.file_signature(train_path)
    hidden_units = selection["selected_parameters"]["hidden_layer_sizes"][0]
    pipeline = _pipeline(hidden_units)
    with tempfile.TemporaryDirectory(
        prefix=".p2_mlp_fit_", dir=run_dir
    ) as temp:
        temp_dir = Path(temp)
        x = y = None
        try:
            x = np.lib.format.open_memmap(
                temp_dir / "x_train.npy",
                mode="w+",
                dtype=np.float32,
                shape=(rows, len(features)),
            )
            y = np.lib.format.open_memmap(
                temp_dir / "y_train.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(rows,),
            )
            position = 0
            observed = np.zeros(2, dtype=np.int64)
            for chunk in pd.read_csv(
                train_path,
                compression="gzip",
                chunksize=CHUNK_ROWS,
                dtype=p2_data.data_dtypes(context),
                float_precision="round_trip",
            ):
                labels = p2_data.validate_chunk(chunk, context, TRAIN_DAYS)
                stop = position + len(chunk)
                if stop > rows:
                    raise ValueError("Train P2 có nhiều dòng hơn split manifest.")
                values = chunk[features].to_numpy(dtype=np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise ValueError(
                        "Train P2 có feature không hữu hạn sau đổi float32."
                    )
                x[position:stop] = values
                y[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != rows or observed.tolist() != [
                int(split_counts[label]) for label in LABELS
            ]:
                raise ValueError("Train P2 không khớp số dòng/nhãn manifest.")
            p2_data.check_file_signature(train_path, signature)
            fit_seconds, n_iter, final_loss, convergence_warnings = _fit(
                pipeline, x, y, run_dir
            )
        finally:
            if x is not None:
                x.flush()
                x._mmap.close()
            if y is not None:
                y.flush()
                y._mmap.close()

        temporary_model = temp_dir / model_path.name
        joblib.dump(pipeline, temporary_model, compress=3)
        summary = {
            "protocol": "P2",
            "source_run_id": context["validation_manifest"]["source_run_id"],
            "validation_run_id": context["validation_manifest"]["validation_run_id"],
            "fit_partition": "Train",
            "train_rows": rows,
            "train_label_counts": split_counts,
            "feature_columns": features,
            "selected_parameters": selection["selected_parameters"],
            "selection_artifact": "p2_mlp_validation_selection.json",
            "sklearn_version": sklearn_version,
            "fit_seconds": fit_seconds,
            "n_iter": n_iter,
            "final_loss": final_loss,
            "convergence_warnings": convergence_warnings,
            "early_stopping": False,
            "test_read_or_used": False,
            "test_features_used_to_filter_train": True,
        }
        with temporary_model.open("rb") as source, model_path.open("xb") as target:
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

    print(f"Đã lưu MLP pipeline P2: {model_path}")
    for message in convergence_warnings:
        print(f"ConvergenceWarning: {message}")
    return model_path


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _read_selection(validation_run_dir)
    run_dir = context["run_dir"]
    model_path = run_dir / "p2_mlp_selected_pipeline.joblib"
    summary_path = run_dir / "p2_mlp_selected_training.json"
    output = run_dir / "p2_mlp_test_metrics.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test MLP P2 đã tồn tại: {output}")
    if not model_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Fit MLP trên Train P2 trước khi đánh giá Test.")

    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    manifest = context["validation_manifest"]
    if (
        summary.get("protocol") != "P2"
        or summary.get("source_run_id") != manifest["source_run_id"]
        or summary.get("validation_run_id") != manifest["validation_run_id"]
        or summary.get("fit_partition") != "Train"
        or summary.get("selected_parameters") != selection["selected_parameters"]
        or summary.get("selection_artifact") != "p2_mlp_validation_selection.json"
        or summary.get("feature_columns") != context["feature_columns"]
        or summary.get("train_rows")
        != sum(
            int(context["split_manifest"]["counts"]["Train"][label])
            for label in LABELS
        )
        or summary.get("sklearn_version") != sklearn_version
        or summary.get("early_stopping") is not False
        or summary.get("test_read_or_used") is not False
        or summary.get("test_features_used_to_filter_train") is not True
    ):
        raise ValueError("Model MLP P2 không khớp lựa chọn hoặc Train.")

    pipeline = joblib.load(model_path)
    if (
        not isinstance(pipeline, Pipeline)
        or list(pipeline.classes_) != [0, 1]
        or list(pipeline.named_steps) != ["scaler", "mlp"]
        or pipeline.named_steps["scaler"].n_features_in_
        != len(context["feature_columns"])
    ):
        raise ValueError("MLP pipeline không khớp nhãn hoặc feature P2.")
    estimator = pipeline.named_steps["mlp"]
    hidden_units = selection["selected_parameters"]["hidden_layer_sizes"]
    if (
        list(estimator.hidden_layer_sizes) != hidden_units
        or any(
            estimator.get_params()[name] != value
            for name, value in selection["selected_parameters"].items()
            if name != "hidden_layer_sizes"
        )
        or pipeline.named_steps["scaler"].n_samples_seen_
        != summary["train_rows"]
    ):
        raise ValueError("Tham số MLP hoặc scaler không khớp Train/Validation.")

    metrics = p2_data.evaluate_partition(pipeline, run_dir, "Test")
    _write_json_new(
        output,
        {
            "protocol": "P2",
            "source_run_id": manifest["source_run_id"],
            "validation_run_id": manifest["validation_run_id"],
            "selected_parameters": selection["selected_parameters"],
            "sklearn_version": sklearn_version,
            "partition": "Test",
            "previously_seen_test": True,
            "test_read_or_used": True,
            "test_features_used_to_filter_train": True,
            "metrics": metrics,
        },
    )
    return pd.DataFrame([metrics])
