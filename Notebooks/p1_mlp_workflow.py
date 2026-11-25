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

import validation_models


CHUNK_ROWS = 20_000
LABELS = ("Normal", "Attack")
CANDIDATE_HIDDEN_UNITS = (32, 64, 128)
FIXED_PARAMETERS = {
    "activation": "relu",
    "alpha": 0.0001,
    "batch_size": 1024,
    "early_stopping": False,
    "learning_rate_init": 0.001,
    "max_iter": 100,
    "random_state": 42,
    "solver": "adam",
}
MIN_AVAILABLE_RAM = 4 * 1024**3
MIN_FREE_DISK = 3 * 1024**3


def current_validation_run_dir(source_run_dir: Path | None = None) -> Path:
    return validation_models.latest_p1_validation_run_dir(source_run_dir)


def _context(validation_run_dir: Path) -> dict:
    return validation_models._load_context(
        validation_run_dir, "P1", current_p1_run=True
    )


def _serialized_parameters(hidden_units: int) -> dict:
    return {
        **FIXED_PARAMETERS,
        "hidden_layer_sizes": [hidden_units],
    }


def _pipeline(hidden_units: int, *, max_iter: int = 100) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    **{**FIXED_PARAMETERS, "max_iter": max_iter},
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
        raise MemoryError("Cần ít nhất 4 GiB RAM khả dụng trước khi fit MLP.")
    if estimated_peak > available * 0.70:
        raise MemoryError(
            "Peak RAM ước tính vượt 70% RAM khả dụng; dừng trước khi fit MLP."
        )
    if free < MIN_FREE_DISK:
        raise OSError("Cần ít nhất 3 GiB đĩa trống trước khi fit MLP.")


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


def prepare_validation_cache(validation_run_dir: Path) -> dict:
    context = _context(validation_run_dir)
    output = context["run_dir"] / "mlp_validation_selection.json"
    if output.exists():
        raise FileExistsError(
            f"Kết quả lựa chọn MLP đã tồn tại; không ghi đè: {output}"
        )
    return validation_models.prepare_training_cache(
        validation_run_dir, "P1", current_p1_run=True
    )


def _compare_validation(
    validation_run_dir: Path, max_iter: int, output_name: str
) -> pd.DataFrame:
    context = _context(validation_run_dir)
    run_dir = context["run_dir"]
    output = run_dir / output_name
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
        or np.bincount(y, minlength=2).tolist()
        != [int(counts[label]) for label in LABELS]
    ):
        raise ValueError("Cache Train nội bộ không khớp manifest.")

    results = []
    try:
        for hidden_units in CANDIDATE_HIDDEN_UNITS:
            print(
                f"Đang fit và đánh giá Validation: "
                f"hidden_layer_sizes=({hidden_units},)"
            )
            pipeline = _pipeline(hidden_units)
            pipeline.set_params(mlp__max_iter=max_iter)
            fit_seconds, n_iter, final_loss, convergence_warnings = _fit(
                pipeline, x, y, run_dir
            )
            metrics = validation_models.evaluate_binary_classifier(
                pipeline, context["validation_path"], context, "Validation"
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
        del x, y
        gc.collect()

    selected = max(
        results, key=lambda row: (row["macro_f1"], -row["hidden_units"])
    )
    selected_parameters = _serialized_parameters(selected["hidden_units"])
    selected_parameters["max_iter"] = max_iter
    _write_json_new(
        output,
        {
            "source_run_id": manifest["source_run_id"],
            "validation_run_id": manifest["validation_run_id"],
            "sklearn_version": sklearn_version,
            "selected_parameters": selected_parameters,
            "selection_rule": (
                "highest macro_f1; on exact tie, fewer hidden units"
            ),
            "early_stopping": False,
            "test_read_or_used": False,
            "results": results,
        },
    )
    print(f"Cấu hình MLP được chọn theo Macro-F1: {selected_parameters}")
    return pd.DataFrame(results)


def compare_validation(validation_run_dir: Path) -> pd.DataFrame:
    return _compare_validation(
        validation_run_dir, 100, "mlp_validation_selection.json"
    )


def compare_convergence_200(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _selected_context(validation_run_dir)
    previous = selection.get("results")
    if (
        not isinstance(previous, list)
        or len(previous) != len(CANDIDATE_HIDDEN_UNITS)
        or {row.get("hidden_units") for row in previous} != set(CANDIDATE_HIDDEN_UNITS)
    ):
        raise ValueError("Kết quả MLP 100 vòng không đủ ba cấu hình để so sánh.")
    validation_models.prepare_training_cache(
        context["run_dir"], "P1", current_p1_run=True
    )
    current = _compare_validation(
        context["run_dir"], 200, "mlp_validation_convergence_200.json"
    )
    print(
        "Khảo sát 200 vòng không tự fit model cuối; "
        "cần xem kết quả trước khi quyết định."
    )
    return pd.concat(
        [
            pd.DataFrame(previous).assign(max_iter=100),
            current.assign(max_iter=200),
        ],
        ignore_index=True,
    )


def _selected_context(
    validation_run_dir: Path, *, convergence_200: bool = False
) -> tuple[dict, dict]:
    context = _context(validation_run_dir)
    selection_name = (
        "mlp_validation_convergence_200.json"
        if convergence_200
        else "mlp_validation_selection.json"
    )
    selection_path = context["run_dir"] / selection_name
    with selection_path.open(encoding="utf-8") as file:
        selection = json.load(file)
    manifest = context["manifest"]
    allowed_parameters = [
        {
            **_serialized_parameters(hidden_units),
            "max_iter": 200 if convergence_200 else 100,
        }
        for hidden_units in CANDIDATE_HIDDEN_UNITS
    ]
    if (
        selection.get("source_run_id") != manifest["source_run_id"]
        or selection.get("validation_run_id") != manifest["validation_run_id"]
        or selection.get("test_read_or_used") is not False
        or selection.get("early_stopping") is not False
        or selection.get("sklearn_version") != sklearn_version
        or selection.get("selected_parameters") not in allowed_parameters
    ):
        raise ValueError("Lựa chọn MLP không khớp run hoặc môi trường.")
    if convergence_200 and selection["selected_parameters"]["hidden_layer_sizes"] != [128]:
        raise ValueError("Selection 200 vòng không khớp cấu hình 128 neuron đã duyệt.")
    return context, selection


def fit_final(validation_run_dir: Path) -> Path:
    context, selection = _selected_context(validation_run_dir, convergence_200=True)
    manifest = context["manifest"]
    source_dir = context["run_dir"].resolve().parent.parent
    train_path = source_dir / "train.csv.gz"
    output = context["run_dir"] / "mlp_selected_pipeline.joblib"
    summary_path = context["run_dir"] / "mlp_selected_training.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError("Model hoặc summary MLP đã tồn tại; không ghi đè.")
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

    matrix_bytes = rows * len(context["feature_columns"]) * np.dtype(np.float32).itemsize
    _check_resources(context["run_dir"], matrix_bytes)
    hidden_units = selection["selected_parameters"]["hidden_layer_sizes"][0]
    pipeline = _pipeline(
        hidden_units, max_iter=selection["selected_parameters"]["max_iter"]
    )
    with tempfile.TemporaryDirectory(
        prefix=".mlp_fit_", dir=context["run_dir"]
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
                Path(temp) / "y.npy", mode="w+", dtype=np.uint8, shape=(rows,)
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
                    raise ValueError("Feature không hữu hạn sau chuyển float32.")
                x[position:stop] = values
                y[position:stop] = labels
                observed += np.bincount(labels, minlength=2)
                position = stop
            if position != rows or observed.tolist() != [
                int(counts[label]) for label in LABELS
            ]:
                raise ValueError("Số dòng hoặc nhãn Train gốc không khớp audit.")
            fit_seconds, n_iter, final_loss, convergence_warnings = _fit(
                pipeline, x, y, context["run_dir"]
            )
        finally:
            if x is not None:
                x.flush()
                x._mmap.close()
            if y is not None:
                y.flush()
                y._mmap.close()

        estimator = pipeline.named_steps["mlp"]
        temporary_model = Path(temp) / "model.joblib"
        joblib.dump(pipeline, temporary_model, compress=3)
        summary = {
            "source_run_id": manifest["source_run_id"],
            "validation_run_id": manifest["validation_run_id"],
            "fit_partition": "Train",
            "train_rows": rows,
            "feature_columns": context["feature_columns"],
            "selected_parameters": selection["selected_parameters"],
            "selection_artifact": "mlp_validation_convergence_200.json",
            "sklearn_version": sklearn_version,
            "fit_seconds": fit_seconds,
            "n_iter": n_iter,
            "final_loss": final_loss,
            "convergence_warnings": convergence_warnings,
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

    print(f"Đã lưu MLP pipeline: {output}")
    return output


def evaluate_test(validation_run_dir: Path) -> pd.DataFrame:
    context, selection = _selected_context(validation_run_dir, convergence_200=True)
    output = context["run_dir"] / "mlp_test_metrics.json"
    model_path = context["run_dir"] / "mlp_selected_pipeline.joblib"
    summary_path = context["run_dir"] / "mlp_selected_training.json"
    if output.exists():
        raise FileExistsError(f"Kết quả Test đã tồn tại: {output}")
    if not model_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Fit MLP trên Train trước khi đánh giá Test.")
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    manifest = context["manifest"]
    if (
        summary.get("source_run_id") != manifest["source_run_id"]
        or summary.get("validation_run_id") != manifest["validation_run_id"]
        or summary.get("selected_parameters") != selection["selected_parameters"]
        or summary.get("selection_artifact") != "mlp_validation_convergence_200.json"
        or summary.get("feature_columns") != context["feature_columns"]
        or summary.get("fit_partition") != "Train"
        or summary.get("train_rows") != manifest["source_train_rows"]
        or summary.get("sklearn_version") != sklearn_version
        or summary.get("early_stopping") is not False
        or summary.get("test_read_or_used") is not False
    ):
        raise ValueError("Model MLP đã fit không khớp cấu hình, run hoặc môi trường.")

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
    pipeline = joblib.load(model_path)
    if (
        not isinstance(pipeline, Pipeline)
        or list(pipeline.classes_) != [0, 1]
        or list(pipeline.named_steps) != ["scaler", "mlp"]
        or pipeline.named_steps["scaler"].n_features_in_
        != len(context["feature_columns"])
    ):
        raise ValueError("MLP pipeline không khớp nhãn hoặc feature.")
    estimator = pipeline.named_steps["mlp"]
    expected_units = selection["selected_parameters"]["hidden_layer_sizes"]
    if (
        list(estimator.hidden_layer_sizes) != expected_units
        or any(
            estimator.get_params()[name] != value
            for name, value in {
                **FIXED_PARAMETERS,
                "max_iter": selection["selected_parameters"]["max_iter"],
            }.items()
        )
    ):
        raise ValueError("Tham số MLP đã lưu không khớp lựa chọn Validation.")

    metrics = validation_models.evaluate_binary_classifier(
        pipeline, test_path, test_context, "Test"
    )
    _write_json_new(
        output,
        {
            "source_run_id": summary["source_run_id"],
            "validation_run_id": summary["validation_run_id"],
            "selected_parameters": summary["selected_parameters"],
            "sklearn_version": sklearn_version,
            "partition": "Test",
            "test_read_or_used": True,
            "metrics": metrics,
        },
    )
    return pd.DataFrame([metrics])
