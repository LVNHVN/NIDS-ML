from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROTOCOLS = {
    "P1": {
        "folder": "P1_Random_Group_Aware",
    },
    "P2": {
        "folder": "P2_Day_Disjoint",
        "source_run_id": "20261109_224942_99497f14",
        "validation_run_id": "20261112_064841_b97fd890",
    },
}
LABEL_TO_CODE = {"Normal": 0, "Attack": 1}
LABELS = ("Normal", "Attack")
EXPECTED_METADATA = ["SourceFile", "CaptureDay", "OriginalLabel", "BinaryLabel"]
CHUNK_ROWS = 20_000


class BinaryPredictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _protocol_config(protocol: str) -> dict[str, str]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol}")
    return PROTOCOLS[protocol]


def _default_run_dir(protocol: str = "P1") -> Path:
    if protocol == "P1":
        return latest_p1_validation_run_dir()
    config = _protocol_config(protocol)
    project_root = Path(__file__).resolve().parents[1]
    return (
        project_root
        / "Dataset"
        / "Processed"
        / config["folder"]
        / config["source_run_id"]
        / "validation_runs"
        / config["validation_run_id"]
    )


def _load_context(
    run_dir: Path | None = None, protocol: str = "P1", *,
    current_p1_run: bool = False,
) -> dict[str, Any]:
    config = _protocol_config(protocol)
    if current_p1_run and (protocol != "P1" or run_dir is None):
        raise ValueError("Current P1 mode requires an explicit P1 Validation run.")
    if run_dir is None and protocol == "P1":
        run_dir = latest_p1_validation_run_dir()
        current_p1_run = True
    else:
        run_dir = Path(run_dir) if run_dir is not None else _default_run_dir(protocol)
        if protocol == "P1":
            current_p1_run = True
    manifest_path = run_dir / "validation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Validation manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    source_run_id = config.get("source_run_id")
    validation_run_id = config.get("validation_run_id")
    if current_p1_run:
        resolved_run = run_dir.resolve()
        source_dir = resolved_run.parent.parent
        p1_root = Path(__file__).resolve().parents[1] / "Dataset" / "Processed" / "P1_Random_Group_Aware"
        if resolved_run.parent.name != "validation_runs" or source_dir.parent != p1_root.resolve():
            raise ValueError("P1 Validation run is outside the P1 split directory.")
        source_run_id = source_dir.name
        validation_run_id = resolved_run.name
    if (
        manifest.get("validation_run_id") != validation_run_id
        or manifest.get("protocol") != protocol
        or manifest.get("source_run_id") != source_run_id
        or manifest.get("test_read_or_used") is not False
        or manifest.get("shared_feature_groups") != 0
    ):
        raise ValueError(
            f"Validation manifest does not match the reviewed {protocol} run."
        )
    if "split_csv_float_precision" in manifest and (
        protocol != "P1" or manifest["split_csv_float_precision"] != "round_trip"
    ):
        raise ValueError("Invalid P1 split CSV float precision.")
    features = manifest.get("feature_columns", [])
    metadata = manifest.get("metadata_columns", [])
    if len(features) != 78 or len(set(features)) != 78:
        raise ValueError(f"{protocol} Validation must contain 78 unique features.")
    if metadata != EXPECTED_METADATA:
        raise ValueError(f"{protocol} Validation metadata schema does not match.")

    internal_path = _run_artifact_path(
        run_dir, manifest.get("train_internal_path"), "train_internal.csv.gz"
    )
    validation_path = _run_artifact_path(
        run_dir, manifest.get("validation_path"), "validation.csv.gz"
    )
    if internal_path.parent.resolve() != run_dir.resolve():
        raise ValueError("Internal Train path is outside the reviewed Validation run.")
    if validation_path.parent.resolve() != run_dir.resolve():
        raise ValueError("Validation path is outside the reviewed Validation run.")

    expected_columns = features + metadata
    for path in (internal_path, validation_path):
        header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
        if header != expected_columns:
            raise ValueError(f"Unexpected columns in {path.name}.")

    for partition in ("TrainInternal", "Validation"):
        rows = manifest.get("partition_rows", {}).get(partition, 0)
        counts = manifest.get("partition_label_counts", {}).get(partition, {})
        if rows <= 0 or sum(counts.get(label, 0) for label in LABELS) != rows:
            raise ValueError(f"Invalid rows or label counts for {partition}.")

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "feature_columns": features,
        "metadata_columns": metadata,
        "columns": expected_columns,
        "internal_path": internal_path,
        "validation_path": validation_path,
    }


def _source_dtypes(context: dict[str, Any]) -> dict[str, Any]:
    return {
        **{column: np.float64 for column in context["feature_columns"]},
        **{column: str for column in context["metadata_columns"]},
    }


def _run_artifact_path(run_dir: Path, stored_path: Any, expected_name: str) -> Path:
    if not isinstance(stored_path, str) or Path(stored_path).name != expected_name:
        raise ValueError(f"Invalid Validation artifact path for {expected_name}.")
    path = run_dir / expected_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing Validation artifact: {path}")
    return path


def latest_p1_validation_run_dir(
    source_run_dir: Path | None = None,
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    p1_root = project_root / "Dataset" / "Processed" / "P1_Random_Group_Aware"
    if not p1_root.is_dir():
        raise FileNotFoundError(f"Missing P1 split directory: {p1_root}")
    expected_source_dir = (
        Path(source_run_dir).resolve() if source_run_dir is not None else None
    )

    candidates = []
    for manifest_path in p1_root.glob(
        "*/validation_runs/*/validation_manifest.json"
    ):
        validation_dir = manifest_path.parent
        source_dir = validation_dir.parent.parent
        if source_dir.parent.resolve() != p1_root.resolve():
            continue
        if (
            expected_source_dir is not None
            and source_dir.resolve() != expected_source_dir
        ):
            continue
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if (
            manifest.get("protocol") == "P1"
            and manifest.get("source_run_id") == source_dir.name
            and manifest.get("validation_run_id") == validation_dir.name
        ):
            candidates.append((source_dir.name, validation_dir.name, validation_dir))

    if not candidates:
        scope = (
            f" for source run {expected_source_dir}"
            if expected_source_dir is not None
            else ""
        )
        raise FileNotFoundError(
            f"No completed P1 Validation run found{scope} under {p1_root}"
        )
    _, _, run_dir = max(
        candidates, key=lambda candidate: (candidate[1], candidate[0])
    )
    context = _load_context(run_dir, "P1", current_p1_run=True)

    source_dir = run_dir.resolve().parent.parent
    split_manifest_path = source_dir / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source split manifest: {split_manifest_path}")
    with split_manifest_path.open(encoding="utf-8") as file:
        split_manifest = json.load(file)
    if (
        split_manifest.get("run_id") != context["manifest"]["source_run_id"]
        or split_manifest.get("feature_columns") != context["feature_columns"]
        or split_manifest.get("shared_feature_groups_train_test") != 0
    ):
        raise ValueError("Latest P1 Validation does not match its audited split.")
    train_counts = split_manifest["split_counts"]["Train"]
    if (
        int(train_counts["rows"]) != int(context["manifest"]["source_train_rows"])
        or any(
            int(train_counts[label])
            != int(context["manifest"]["source_train_label_counts"][label])
            for label in LABELS
        )
    ):
        raise ValueError("Validation source Train counts do not match split audit.")
    return context["run_dir"]


def _validate_chunk(
    chunk: pd.DataFrame, context: dict[str, Any]
) -> np.ndarray:
    if chunk.columns.tolist() != context["columns"]:
        raise ValueError("Validation CSV schema changed.")
    metadata = context["metadata_columns"]
    if chunk[metadata].isna().any().any():
        raise ValueError("Metadata contains missing values.")
    if chunk[metadata].apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError("Metadata contains empty values.")
    codes = chunk["BinaryLabel"].map(LABEL_TO_CODE)
    if codes.isna().any():
        raise ValueError("BinaryLabel contains an unsupported value.")
    values = chunk[context["feature_columns"]].to_numpy(
        dtype=np.float64, copy=False
    )
    if not np.isfinite(values).all():
        raise ValueError("Feature data contains NaN or Infinity.")
    return codes.to_numpy(dtype=np.uint8, copy=False)


def evaluate_binary_classifier(
    model: BinaryPredictor, path: Path, context: dict, partition: str
) -> dict:
    manifest = context["manifest"]
    expected_rows = int(manifest["partition_rows"][partition])
    expected_counts = manifest["partition_label_counts"][partition]
    confusion = np.zeros((2, 2), dtype=np.int64)
    observed = np.zeros(2, dtype=np.int64)
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=CHUNK_ROWS,
        dtype=_source_dtypes(context),
        float_precision=manifest.get("split_csv_float_precision"),
    ):
        labels = _validate_chunk(chunk, context)
        values = chunk[context["feature_columns"]].to_numpy(
            dtype=np.float32, copy=False
        )
        if not np.isfinite(values).all():
            raise ValueError("Chuyển feature sang float32 tạo giá trị không hữu hạn.")
        predictions = model.predict(values)
        confusion += confusion_matrix(labels, predictions, labels=[0, 1])
        observed += np.bincount(labels, minlength=2)

    expected = [int(expected_counts[label]) for label in LABELS]
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


def prepare_training_cache(
    run_dir: Path | None = None, protocol: str = "P1", *,
    current_p1_run: bool = False,
) -> dict[str, Any]:
    context = _load_context(run_dir, protocol, current_p1_run=current_p1_run)
    run_dir = context["run_dir"]
    manifest = context["manifest"]
    train_rows = int(manifest["partition_rows"]["TrainInternal"])
    expected_labels = manifest["partition_label_counts"]["TrainInternal"]
    feature_columns = context["feature_columns"]
    cache_dir = run_dir / "training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"

    if cache_dir.exists():
        if not x_path.is_file() or not y_path.is_file():
            raise FileExistsError(
                f"Incomplete training cache exists and will not be overwritten: {cache_dir}"
            )
        x_train = np.load(x_path, mmap_mode="r")
        y_train = np.load(y_path, mmap_mode="r")
        if (
            x_train.shape != (train_rows, len(feature_columns))
            or x_train.dtype != np.float32
            or y_train.shape != (train_rows,)
            or y_train.dtype != np.uint8
        ):
            raise ValueError("Existing training cache has an unexpected shape or dtype.")
        observed_labels = np.bincount(y_train, minlength=2)
        if observed_labels.tolist() != [
            int(expected_labels[label]) for label in LABELS
        ]:
            raise ValueError("Existing training cache label counts do not match.")
        del x_train, y_train
        gc.collect()
        print(f"Using validated training cache: {cache_dir}")
        return context

    matrix_bytes = train_rows * len(feature_columns) * np.dtype(np.float32).itemsize
    disk_required = int(matrix_bytes + train_rows + 64 * 1024**2)
    if shutil.disk_usage(run_dir).free < disk_required:
        raise OSError("Insufficient disk space for the float32 training cache.")
    working_bytes = CHUNK_ROWS * len(context["columns"]) * 8 * 2 + 2 * 1024**2
    available_bytes = psutil.virtual_memory().available
    if working_bytes > available_bytes * 0.5:
        raise MemoryError("Insufficient RAM for chunked training-cache preparation.")

    temp_dir = Path(tempfile.mkdtemp(prefix=".training_cache_tmp_", dir=run_dir))
    x_temp = temp_dir / x_path.name
    y_temp = temp_dir / y_path.name
    x_map = None
    y_map = None
    try:
        x_map = np.lib.format.open_memmap(
            x_temp,
            mode="w+",
            dtype=np.float32,
            shape=(train_rows, len(feature_columns)),
        )
        y_map = np.lib.format.open_memmap(
            y_temp, mode="w+", dtype=np.uint8, shape=(train_rows,)
        )
        row_position = 0
        observed_labels = np.zeros(2, dtype=np.int64)
        for chunk in pd.read_csv(
            context["internal_path"],
            compression="gzip",
            chunksize=CHUNK_ROWS,
            dtype=_source_dtypes(context),
            float_precision=context["manifest"].get("split_csv_float_precision"),
        ):
            codes = _validate_chunk(chunk, context)
            stop = row_position + len(chunk)
            if stop > train_rows:
                raise AssertionError("Internal Train contains more rows than manifest.")
            values = chunk[feature_columns].to_numpy(
                dtype=np.float32, copy=False
            )
            if not np.isfinite(values).all():
                raise ValueError("Feature conversion to float32 produced non-finite values.")
            x_map[row_position:stop] = values
            y_map[row_position:stop] = codes
            observed_labels += np.bincount(codes, minlength=2)
            row_position = stop
        if row_position != train_rows:
            raise AssertionError("Internal Train row count does not match manifest.")
        if observed_labels.tolist() != [
            int(expected_labels[label]) for label in LABELS
        ]:
            raise AssertionError("Internal Train label counts do not match manifest.")
        x_map.flush()
        y_map.flush()
        del x_map, y_map
        x_map = None
        y_map = None
        os.replace(temp_dir, cache_dir)
    finally:
        if x_map is not None:
            del x_map
        if y_map is not None:
            del y_map
        gc.collect()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print(
        f"Prepared {train_rows:,} internal Train rows x "
        f"{len(feature_columns)} features."
    )
    print(f"Normal: {expected_labels['Normal']:,}; Attack: {expected_labels['Attack']:,}")
    print(f"Cache saved to: {cache_dir}")
    return context


def fit_baselines(
    run_dir: Path | None = None,
    *,
    approved: bool = False,
    protocol: str = "P1",
) -> pd.DataFrame:
    if approved is not True:
        raise PermissionError("Set the explicit Validation approval flag before fitting.")
    context = _load_context(run_dir, protocol)
    manifest = context["manifest"]
    run_dir = context["run_dir"]
    cache_dir = run_dir / "training_cache"
    x_path = cache_dir / "X_train_internal_float32.npy"
    y_path = cache_dir / "y_train_internal_uint8.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError("Run prepare_training_cache before fitting models.")

    train_rows = int(manifest["partition_rows"]["TrainInternal"])
    feature_columns = context["feature_columns"]
    x_train = np.load(x_path, mmap_mode="r")
    y_train = np.load(y_path, mmap_mode="r")
    if (
        x_train.shape != (train_rows, len(feature_columns))
        or y_train.shape != (train_rows,)
    ):
        raise ValueError("Training cache does not match the reviewed manifest.")

    specs = [
        {
            "name": "Logistic Regression",
            "model_file": "validation_logistic_regression_pipeline.joblib",
            "summary_file": "validation_logistic_regression_training_summary.json",
            "parameters": {
                "solver": "saga",
                "C": 1.0,
                "max_iter": 200,
                "tol": 1e-3,
                "class_weight": None,
                "random_state": 42,
            },
            "factory": lambda parameters: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logistic_regression", LogisticRegression(**parameters)),
                ]
            ),
            "peak_bytes": int(
                x_train.nbytes * 4.0
            ),
        },
        {
            "name": "Decision Tree",
            "model_file": "validation_decision_tree_model.joblib",
            "summary_file": "validation_decision_tree_training_summary.json",
            "parameters": {
                "criterion": "gini",
                "max_depth": 20,
                "min_samples_leaf": 10,
                "max_leaf_nodes": 50_000,
                "class_weight": None,
                "random_state": 42,
            },
            "factory": lambda parameters: DecisionTreeClassifier(**parameters),
            "peak_bytes": int(x_train.nbytes * 3.0 + 512 * 1024**2),
        },
        {
            "name": "Random Forest",
            "model_file": "validation_random_forest_model.joblib",
            "summary_file": "validation_random_forest_training_summary.json",
            "parameters": {
                "n_estimators": 50,
                "criterion": "gini",
                "max_depth": 16,
                "min_samples_leaf": 20,
                "max_leaf_nodes": 10_000,
                "max_features": "sqrt",
                "bootstrap": True,
                "max_samples": 0.25,
                "class_weight": None,
                "random_state": 42,
                "n_jobs": 1,
                "verbose": 0,
            },
            "factory": lambda parameters: RandomForestClassifier(**parameters),
            "peak_bytes": int(
                x_train.nbytes * 3.0
                + 50 * (2 * 10_000 - 1) * 128
                + 512 * 1024**2
            ),
        },
    ]
    output_paths = [
        run_dir / name
        for spec in specs
        for name in (
            spec["model_file"],
            spec["model_file"].replace(".joblib", ".tmp.joblib"),
            spec["summary_file"],
            spec["summary_file"].replace(".json", ".tmp.json"),
        )
    ]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            f"Validation model artifacts already exist and will not be overwritten: {existing_paths}"
        )

    summaries = []
    try:
        for spec in specs:
            available_bytes = psutil.virtual_memory().available
            print(
                f"{spec['name']}: estimated peak "
                f"{spec['peak_bytes'] / 1024**3:.2f} GiB; "
                f"available RAM {available_bytes / 1024**3:.2f} GiB."
            )
            if spec["peak_bytes"] > available_bytes * 0.75:
                raise MemoryError(
                    f"Insufficient RAM margin for {spec['name']}; model was not fit."
                )
            model = spec["factory"](spec["parameters"])
            started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            if list(model.classes_) != [0, 1]:
                raise AssertionError(f"{spec['name']} learned unexpected classes.")

            model_path = run_dir / spec["model_file"]
            model_temp_path = run_dir / spec["model_file"].replace(
                ".joblib", ".tmp.joblib"
            )
            joblib.dump(model, model_temp_path, compress=3)
            os.replace(model_temp_path, model_path)
            summary = {
                "validation_run_id": manifest["validation_run_id"],
                "source_run_id": manifest["source_run_id"],
                "fit_partition": "TrainInternal",
                "fit_rows": train_rows,
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
                "label_mapping": LABEL_TO_CODE,
                "model": spec["name"],
                "model_parameters": spec["parameters"],
                "fit_seconds": fit_seconds,
                "model_path": str(model_path),
            }
            if spec["name"] == "Logistic Regression":
                summary["n_iter"] = model.named_steps[
                    "logistic_regression"
                ].n_iter_.tolist()
                summary["scaler_fit_partition"] = "TrainInternal"
            elif spec["name"] == "Decision Tree":
                summary["tree_depth_observed"] = int(model.get_depth())
                summary["tree_leaf_count"] = int(model.get_n_leaves())
            else:
                summary["tree_count_observed"] = len(model.estimators_)
                summary["max_depth_observed"] = max(
                    int(tree.get_depth()) for tree in model.estimators_
                )
            summary_path = run_dir / spec["summary_file"]
            summary_temp_path = run_dir / spec["summary_file"].replace(
                ".json", ".tmp.json"
            )
            with summary_temp_path.open("w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
            os.replace(summary_temp_path, summary_path)
            summaries.append(
                {
                    "Model": spec["name"],
                    "FitSeconds": fit_seconds,
                    "ModelPath": str(model_path),
                }
            )
            print(f"{spec['name']} fit completed in {fit_seconds:.1f} seconds.")
            del model
            gc.collect()
    finally:
        del x_train, y_train
        gc.collect()

    result = pd.DataFrame(summaries)
    print("All baseline models were fit on TrainInternal only.")
    return result


def evaluate_baselines(
    run_dir: Path | None = None, protocol: str = "P1"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = _load_context(run_dir, protocol)
    manifest = context["manifest"]
    run_dir = context["run_dir"]
    model_specs = [
        ("Logistic Regression", "validation_logistic_regression_pipeline.joblib"),
        ("Decision Tree", "validation_decision_tree_model.joblib"),
        ("Random Forest", "validation_random_forest_model.joblib"),
    ]
    models = {}
    for model_name, model_filename in model_specs:
        model_path = run_dir / model_filename
        summary_path = run_dir / model_filename.replace(
            "_pipeline.joblib", "_training_summary.json"
        ).replace("_model.joblib", "_training_summary.json")
        if not model_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing fitted Validation model: {model_path}")
        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)
        if (
            summary.get("validation_run_id") != manifest["validation_run_id"]
            or summary.get("source_run_id") != manifest["source_run_id"]
            or summary.get("fit_partition") != "TrainInternal"
            or summary.get("feature_columns") != context["feature_columns"]
        ):
            raise ValueError(f"Invalid training summary for {model_name}.")
        model = joblib.load(model_path)
        if list(model.classes_) != [0, 1]:
            raise ValueError(f"{model_name} classes do not match Normal=0/Attack=1.")
        models[model_name] = model

    metric_path = run_dir / "validation_model_comparison.csv"
    confusion_path = run_dir / "validation_confusion_matrices.csv"
    summary_path = run_dir / "validation_evaluation_summary.json"
    output_paths = [
        metric_path,
        metric_path.with_suffix(".tmp.csv"),
        confusion_path,
        confusion_path.with_suffix(".tmp.csv"),
        summary_path,
        summary_path.with_suffix(".tmp.json"),
    ]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            f"Validation evaluation artifacts already exist and will not be overwritten: {existing_paths}"
        )

    confusions = {
        model_name: np.zeros((2, 2), dtype=np.int64) for model_name in models
    }
    observed_rows = 0
    observed_label_counts = np.zeros(2, dtype=np.int64)
    for chunk in pd.read_csv(
        context["validation_path"],
        compression="gzip",
        chunksize=CHUNK_ROWS,
        dtype=_source_dtypes(context),
        float_precision=context["manifest"].get("split_csv_float_precision"),
    ):
        y_true = _validate_chunk(chunk, context)
        observed_label_counts += np.bincount(y_true, minlength=2)
        values = chunk[context["feature_columns"]].to_numpy(
            dtype=np.float32, copy=False
        )
        if not np.isfinite(values).all():
            raise ValueError("Feature conversion to float32 produced non-finite values.")
        for model_name, model in models.items():
            predicted = model.predict(values).astype(np.uint8, copy=False)
            encoded = y_true.astype(np.int64) * 2 + predicted.astype(np.int64)
            confusions[model_name] += np.bincount(
                encoded, minlength=4
            ).reshape(2, 2)
        observed_rows += len(chunk)

    expected_rows = int(manifest["partition_rows"]["Validation"])
    expected_labels = manifest["partition_label_counts"]["Validation"]
    if observed_rows != expected_rows:
        raise AssertionError("Validation row count does not match manifest.")
    if observed_label_counts.tolist() != [
        int(expected_labels[label]) for label in LABELS
    ]:
        raise AssertionError("Validation label counts do not match manifest.")

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    metric_rows = []
    confusion_rows = []
    confusion_json = {}
    for model_name, _ in model_specs:
        tn, fp, fn, tp = (int(value) for value in confusions[model_name].ravel())
        total = tn + fp + fn + tp
        precision_attack = divide(tp, tp + fp)
        recall_attack = divide(tp, tp + fn)
        f1_attack = divide(2 * tp, 2 * tp + fp + fn)
        precision_normal = divide(tn, tn + fn)
        recall_normal = divide(tn, tn + fp)
        f1_normal = divide(2 * tn, 2 * tn + fp + fn)
        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": divide(tn + tp, total),
                "Precision_Attack": precision_attack,
                "Recall_Attack": recall_attack,
                "F1_Attack": f1_attack,
                "Macro_F1": (f1_attack + f1_normal) / 2,
                "FPR": divide(fp, fp + tn),
                "FNR": divide(fn, fn + tp),
                "ValidationRows": total,
            }
        )
        confusion_rows.append(
            {
                "Model": model_name,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "ValidationRows": total,
            }
        )
        confusion_json[model_name] = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

    metrics = pd.DataFrame(metric_rows)
    confusion_table = pd.DataFrame(confusion_rows)
    metrics.to_csv(metric_path.with_suffix(".tmp.csv"), index=False, encoding="utf-8")
    confusion_table.to_csv(
        confusion_path.with_suffix(".tmp.csv"), index=False, encoding="utf-8"
    )
    evaluation_summary = {
        "validation_run_id": manifest["validation_run_id"],
        "source_run_id": manifest["source_run_id"],
        "evaluated_partition": "Validation",
        "validation_rows": observed_rows,
        "validation_label_counts": {
            label: int(observed_label_counts[code])
            for code, label in enumerate(LABELS)
        },
        "test_read_or_used": False,
        "metrics": metric_rows,
        "confusion_matrices": confusion_json,
    }
    with summary_path.with_suffix(".tmp.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation_summary, file, ensure_ascii=False, indent=2)
    os.replace(metric_path.with_suffix(".tmp.csv"), metric_path)
    os.replace(confusion_path.with_suffix(".tmp.csv"), confusion_path)
    os.replace(summary_path.with_suffix(".tmp.json"), summary_path)
    print(f"Validation evaluation saved to: {run_dir}")
    return metrics, confusion_table
