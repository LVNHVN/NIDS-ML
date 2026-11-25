from __future__ import annotations

from collections import Counter
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import GroupShuffleSplit


EXPECTED_METADATA = ["SourceFile", "CaptureDay", "OriginalLabel", "BinaryLabel"]
LABEL_TO_CODE = {"Normal": 0, "Attack": 1}
LABELS = ("Normal", "Attack")
VALIDATION_FRACTION = 0.20
VALIDATION_RANDOM_STATE = 42
CHUNK_ROWS = 20_000


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _relative_to_run(path: Path, run_dir: Path) -> str:
    return Path(os.path.relpath(path, start=run_dir)).as_posix()


def _validate_results(
    source_run_dir: Path,
    result_specs: list[tuple[str, str, Path]],
    source_run_id: str,
    train_rows: int,
    test_rows: int,
    feature_columns: list[str],
) -> None:
    for result_filename, model_path_key, expected_model_path in result_specs:
        result_path = source_run_dir / result_filename
        if not result_path.is_file():
            raise FileNotFoundError(f"Thiếu artifact kết quả model: {result_path}")
        result = _read_json(result_path)
        if (
            result.get("run_id") != source_run_id
            or result.get("train_rows") != train_rows
            or result.get("test_rows") != test_rows
            or result.get("feature_columns") != feature_columns
        ):
            raise ValueError(f"Artifact {result_filename} không khớp manifest split.")
        if Path(result.get(model_path_key, "")).resolve() != expected_model_path.resolve():
            raise ValueError(f"Đường dẫn model trong {result_filename} không khớp run.")
        if not expected_model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy model đã lưu: {expected_model_path}")


def _latest_p1_split_run_dir(project_root: Path) -> Path:
    p1_root = project_root / "Dataset" / "Processed" / "P1_Random_Group_Aware"
    if not p1_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục split P1: {p1_root}")
    candidates = [
        run_dir
        for run_dir in p1_root.iterdir()
        if run_dir.is_dir() and (run_dir / "split_manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy split P1 hoàn chỉnh trong {p1_root}")
    return max(candidates, key=lambda run_dir: run_dir.name)


def _load_protocol(
    protocol: str, project_root: Path, p1_run_dir: Path | None = None
) -> dict:
    if protocol == "P1":
        p1_root = project_root / "Dataset" / "Processed" / "P1_Random_Group_Aware"
        source_run_dir = (
            Path(p1_run_dir).resolve()
            if p1_run_dir is not None
            else _latest_p1_split_run_dir(project_root)
        )
        if source_run_dir.resolve().parent != p1_root.resolve():
            raise ValueError("Run P1 phải nằm trực tiếp trong thư mục P1.")
        source_run_id = source_run_dir.name
        manifest_path = source_run_dir / "split_manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("run_id") != source_run_id
            or manifest.get("split_method") not in (
                "feature-group hash threshold",
                "seeded feature-group hash threshold",
            )
            or manifest.get("shared_feature_groups_train_test") != 0
        ):
            raise ValueError("Manifest không khớp run P1 đã được đánh giá.")
        feature_columns = manifest.get("feature_columns", [])
        metadata_columns = manifest.get("metadata_columns_excluded_from_model", [])
        train_counts = manifest.get("split_counts", {}).get("Train", {})
        test_counts = manifest.get("split_counts", {}).get("Test", {})
        train_rows = int(train_counts.get("rows", 0))
        test_rows = int(test_counts.get("rows", 0))
        expected_label_counts = {
            label: int(train_counts.get(label, 0)) for label in LABELS
        }
        train_path = source_run_dir / "train.csv.gz"
        test_path = source_run_dir / "test.csv.gz"
        hash_key = manifest.get("hash_key")
        result_specs = [
            (
                "experiment_results.json",
                "pipeline_path",
                source_run_dir / "logistic_regression_pipeline.joblib",
            ),
            (
                "decision_tree_results.json",
                "model_path",
                source_run_dir / "decision_tree_model.joblib",
            ),
            (
                "random_forest_results.json",
                "model_path",
                source_run_dir / "random_forest_model.joblib",
            ),
        ]
    elif protocol == "P2":
        if p1_run_dir is not None:
            raise ValueError("p1_run_dir chỉ áp dụng cho P1.")
        source_run_id = "20261109_224942_99497f14"
        source_run_dir = (
            project_root
            / "Dataset"
            / "Processed"
            / "P2_Day_Disjoint"
            / source_run_id
        )
        manifest_path = source_run_dir / "p2_filtered_split_manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("run_id") != source_run_id
            or manifest.get("split_method") != "Day-Disjoint"
            or manifest.get("shared_feature_groups_after_filtering") != 0
            or manifest.get("overlap_policy")
            != "keep_all_test_and_remove_train_rows_whose_feature_hash_occurs_in_test"
        ):
            raise ValueError("Manifest không khớp run P2 đã được đánh giá.")
        feature_columns = manifest.get("feature_columns", [])
        metadata_columns = manifest.get("metadata_columns", [])
        train_rows = int(manifest.get("final_train_rows", 0))
        test_rows = int(manifest.get("final_test_rows", 0))
        source_labels = manifest.get("train_label_counts", {})
        removed_labels = manifest.get("removed_train_label_counts", {})
        expected_label_counts = {
            label: int(source_labels.get(label, 0)) - int(removed_labels.get(label, 0))
            for label in LABELS
        }
        train_path = Path(manifest.get("train_path", "")).resolve()
        test_path = Path(manifest.get("test_path", "")).resolve()
        hash_key = f"{42:016x}"
        result_specs = [
            (
                "p2_logistic_regression_results.json",
                "model_path",
                source_run_dir / "p2_logistic_regression_model.joblib",
            ),
            (
                "p2_decision_tree_results.json",
                "model_path",
                source_run_dir / "p2_decision_tree_model.joblib",
            ),
            (
                "p2_random_forest_results.json",
                "model_path",
                source_run_dir / "p2_random_forest_model.joblib",
            ),
        ]
    else:
        raise ValueError("Protocol phải là 'P1' hoặc 'P2'.")

    if not source_run_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy run nguồn {protocol}: {source_run_dir}")
    if len(feature_columns) != 78 or len(set(feature_columns)) != 78:
        raise ValueError(f"Danh sách feature {protocol} không gồm 78 cột duy nhất.")
    if metadata_columns != EXPECTED_METADATA:
        raise ValueError(f"Metadata {protocol} không khớp schema đã duyệt.")
    if train_rows <= 0 or test_rows <= 0:
        raise ValueError(f"Counts Train/Test của {protocol} không hợp lệ.")
    if sum(expected_label_counts.values()) != train_rows:
        raise ValueError(f"Counts nhãn Train {protocol} không khớp số dòng.")
    p1_default_hash = (
        protocol == "P1"
        and manifest.get("split_method") == "feature-group hash threshold"
        and "hash_key" not in manifest
    )
    if not p1_default_hash and hash_key != f"{42:016x}":
        raise ValueError(f"Hash key {protocol} không khớp split đã lưu.")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file split nguồn {protocol}.")
    if protocol == "P2" and train_path != (source_run_dir / "p2_train.csv.gz").resolve():
        raise ValueError("Đường dẫn Train P2 trong manifest không nằm trong run đã duyệt.")
    if protocol == "P2" and test_path != (source_run_dir / "p2_test.csv.gz").resolve():
        raise ValueError("Đường dẫn Test P2 trong manifest không nằm trong run đã duyệt.")

    if protocol == "P2" and p1_run_dir is None:
        _validate_results(
            source_run_dir,
            result_specs,
            source_run_id,
            train_rows,
            test_rows,
            feature_columns,
        )
    return {
        "source_run_id": source_run_id,
        "source_run_dir": source_run_dir,
        "source_manifest_path": manifest_path,
        "source_manifest": manifest,
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "expected_label_counts": expected_label_counts,
        "feature_columns": feature_columns,
        "metadata_columns": metadata_columns,
        "hash_key": hash_key,
    }


def latest_p1_split_run_dir(project_root: Path) -> Path:
    source_run_dir = _latest_p1_split_run_dir(project_root)
    _load_protocol("P1", project_root, p1_run_dir=source_run_dir)
    return source_run_dir


def _validate_chunk(
    chunk: pd.DataFrame,
    expected_columns: list[str],
    feature_columns: list[str],
    metadata_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if chunk.columns.tolist() != expected_columns:
        raise ValueError("Schema chunk Train không khớp header.")
    if chunk[metadata_columns].isna().any().any():
        raise ValueError("Train có metadata bị thiếu.")
    if chunk[metadata_columns].apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError("Train có metadata rỗng.")
    label_codes = chunk["BinaryLabel"].map(LABEL_TO_CODE)
    if label_codes.isna().any():
        raise ValueError("Train có BinaryLabel ngoài Normal/Attack.")
    feature_values = chunk.loc[:, feature_columns].to_numpy(
        dtype=np.float64, copy=False
    )
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature Train chứa NaN hoặc Infinity.")
    group_hashes = pd.util.hash_pandas_object(
        chunk.loc[:, feature_columns],
        index=False,
        hash_key=VALIDATION_HASH_KEY,
    ).to_numpy(dtype=np.uint64, copy=False)
    return group_hashes, label_codes.to_numpy(dtype=np.uint8, copy=False)


VALIDATION_HASH_KEY = f"{42:016x}"


def create_internal_validation_split(
    protocol: str, *, p1_run_dir: Path | None = None
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    source = _load_protocol(protocol, project_root, p1_run_dir)
    source_run_dir = source["source_run_dir"]
    source_manifest = source["source_manifest"]
    float_precision = source_manifest.get("split_csv_float_precision") if protocol == "P1" else None
    if (
        protocol == "P1"
        and "split_csv_float_precision" in source_manifest
        and float_precision != "round_trip"
    ):
        raise ValueError("Chế độ đọc số thực của split P1 không hợp lệ.")
    train_path = source["train_path"]
    feature_columns = source["feature_columns"]
    metadata_columns = source["metadata_columns"]
    train_rows = source["train_rows"]
    expected_label_counts = source["expected_label_counts"]

    expected_columns = pd.read_csv(
        train_path, compression="gzip", nrows=0
    ).columns.tolist()
    if (
        len(expected_columns) != 82
        or set(expected_columns) != set(feature_columns + metadata_columns)
    ):
        raise ValueError(f"Header Train {protocol} không khớp manifest.")
    feature_dtypes = {column: np.float64 for column in feature_columns}
    metadata_dtypes = {column: str for column in metadata_columns}
    source_dtypes = {**feature_dtypes, **metadata_dtypes}

    available_bytes = psutil.virtual_memory().available
    estimated_working_bytes = train_rows * 48 + CHUNK_ROWS * 82 * 8 * 3
    if estimated_working_bytes > available_bytes * 0.5:
        raise MemoryError(
            f"RAM khả dụng không đủ biên an toàn cho split {protocol}: "
            f"ước tính {estimated_working_bytes / 1024**3:.2f} GiB."
        )
    free_disk_bytes = shutil.disk_usage(source_run_dir).free
    estimated_disk_bytes = max(train_path.stat().st_size * 2, 512 * 1024**2)
    if free_disk_bytes < estimated_disk_bytes:
        raise OSError(
            f"Đĩa trống không đủ cho split Validation {protocol}: "
            f"cần khoảng {estimated_disk_bytes / 1024**3:.2f} GiB."
        )

    source_stat = train_path.stat()
    group_parts = []
    label_parts = []
    observed_rows = 0
    observed_label_counts = Counter()
    for chunk in pd.read_csv(
        train_path,
        compression="gzip",
        chunksize=CHUNK_ROWS,
        dtype=source_dtypes,
        float_precision=float_precision,
    ):
        group_hashes, label_codes = _validate_chunk(
            chunk, expected_columns, feature_columns, metadata_columns
        )
        group_parts.append(group_hashes)
        label_parts.append(label_codes)
        observed_rows += len(chunk)
        observed_label_counts.update(
            {
                LABELS[code]: int(count)
                for code, count in enumerate(np.bincount(label_codes, minlength=2))
                if count
            }
        )
    if (train_path.stat().st_size, train_path.stat().st_mtime_ns) != (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ):
        raise RuntimeError("File Train nguồn thay đổi trong lúc audit.")
    if observed_rows != train_rows or dict(observed_label_counts) != expected_label_counts:
        raise AssertionError("Số dòng/nhãn Train quan sát được không khớp manifest.")

    group_ids = np.concatenate(group_parts)
    label_codes = np.concatenate(label_parts)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=VALIDATION_RANDOM_STATE,
    )
    internal_indices, validation_indices = next(
        splitter.split(group_ids, label_codes, groups=group_ids)
    )
    validation_group_ids = np.unique(group_ids[validation_indices])
    internal_group_ids = np.unique(group_ids[internal_indices])
    shared_groups = np.intersect1d(
        internal_group_ids, validation_group_ids, assume_unique=True
    )
    if shared_groups.size:
        raise AssertionError("Có feature group giao giữa Train nội bộ và Validation.")

    expected_partition_counts = {
        "TrainInternal": np.bincount(label_codes[internal_indices], minlength=2),
        "Validation": np.bincount(label_codes[validation_indices], minlength=2),
    }
    partition_rows = {
        "TrainInternal": int(len(internal_indices)),
        "Validation": int(len(validation_indices)),
    }
    validation_root = source_run_dir / "validation_runs"
    validation_root.mkdir(parents=True, exist_ok=True)
    validation_run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    final_run_dir = validation_root / validation_run_id
    if final_run_dir.exists():
        raise FileExistsError(f"Validation run đã tồn tại: {final_run_dir}")

    temp_run_dir = Path(
        tempfile.mkdtemp(prefix=".validation_tmp_", dir=validation_root)
    )
    try:
        internal_temp_path = temp_run_dir / "train_internal.csv.gz"
        validation_temp_path = temp_run_dir / "validation.csv.gz"
        streamed_rows = {"TrainInternal": 0, "Validation": 0}
        streamed_label_counts = {
            name: np.zeros(2, dtype=np.int64) for name in streamed_rows
        }
        day_label_counts = Counter()
        original_label_counts = Counter()
        write_headers = {"TrainInternal": True, "Validation": True}

        with gzip.open(
            internal_temp_path, "wt", encoding="utf-8", newline=""
        ) as internal_file, gzip.open(
            validation_temp_path, "wt", encoding="utf-8", newline=""
        ) as validation_file:
            output_files = {
                "TrainInternal": internal_file,
                "Validation": validation_file,
            }
            for chunk in pd.read_csv(
                train_path,
                compression="gzip",
                chunksize=CHUNK_ROWS,
                dtype=source_dtypes,
                float_precision=float_precision,
            ):
                group_hashes, chunk_labels = _validate_chunk(
                    chunk, expected_columns, feature_columns, metadata_columns
                )
                validation_mask = np.isin(
                    group_hashes, validation_group_ids, assume_unique=False
                )
                partition_masks = {
                    "TrainInternal": ~validation_mask,
                    "Validation": validation_mask,
                }
                for partition, mask in partition_masks.items():
                    partition_chunk = chunk.loc[mask]
                    if partition_chunk.empty:
                        continue
                    partition_chunk.to_csv(
                        output_files[partition],
                        index=False,
                        header=write_headers[partition],
                    )
                    write_headers[partition] = False
                    streamed_rows[partition] += len(partition_chunk)
                    streamed_label_counts[partition] += np.bincount(
                        chunk_labels[mask], minlength=2
                    )
                    for (day, label), count in partition_chunk.groupby(
                        ["CaptureDay", "BinaryLabel"], sort=False
                    ).size().items():
                        day_label_counts[
                            (partition, str(day), str(label))
                        ] += int(count)
                    for (original_label, label), count in partition_chunk.groupby(
                        ["OriginalLabel", "BinaryLabel"], sort=False
                    ).size().items():
                        original_label_counts[
                            (partition, str(original_label), str(label))
                        ] += int(count)

        if (train_path.stat().st_size, train_path.stat().st_mtime_ns) != (
            source_stat.st_size,
            source_stat.st_mtime_ns,
        ):
            raise RuntimeError("File Train nguồn thay đổi trong lúc tạo Validation.")
        for partition in ("TrainInternal", "Validation"):
            expected_labels = expected_partition_counts[partition]
            if (
                streamed_rows[partition] != partition_rows[partition]
                or not np.array_equal(streamed_label_counts[partition], expected_labels)
                or write_headers[partition]
            ):
                raise AssertionError(
                    f"Rows/nhãn ghi ra không khớp fold {partition}."
                )
        if sum(streamed_rows.values()) != train_rows:
            raise AssertionError("Train nội bộ + Validation không bảo toàn số dòng.")

        train_internal_final = final_run_dir / "train_internal.csv.gz"
        validation_final = final_run_dir / "validation.csv.gz"
        partition_audit = pd.DataFrame(
            [
                {
                    "Partition": partition,
                    "Rows": streamed_rows[partition],
                    "Normal": int(streamed_label_counts[partition][0]),
                    "Attack": int(streamed_label_counts[partition][1]),
                    "AttackRatePercent": (
                        100
                        * streamed_label_counts[partition][1]
                        / streamed_rows[partition]
                    ),
                    "UniqueFeatureGroups": int(
                        len(
                            internal_group_ids
                            if partition == "TrainInternal"
                            else validation_group_ids
                        )
                    ),
                }
                for partition in ("TrainInternal", "Validation")
            ]
        )
        day_label_audit = pd.DataFrame(
            [
                {
                    "Partition": partition,
                    "CaptureDay": day,
                    "BinaryLabel": label,
                    "Rows": count,
                }
                for (partition, day, label), count in sorted(day_label_counts.items())
            ]
        )
        original_label_audit = pd.DataFrame(
            [
                {
                    "Partition": partition,
                    "OriginalLabel": original_label,
                    "BinaryLabel": label,
                    "Rows": count,
                }
                for (partition, original_label, label), count
                in sorted(original_label_counts.items())
            ]
        )
        partition_audit.to_csv(
            temp_run_dir / "partition_audit.csv", index=False, encoding="utf-8"
        )
        day_label_audit.to_csv(
            temp_run_dir / "day_label_audit.csv", index=False, encoding="utf-8"
        )
        original_label_audit.to_csv(
            temp_run_dir / "original_label_audit.csv",
            index=False,
            encoding="utf-8",
        )
        manifest = {
            "validation_run_id": validation_run_id,
            "protocol": protocol,
            "source_run_id": source["source_run_id"],
            "source_manifest_path": _relative_to_run(
                source["source_manifest_path"], final_run_dir
            ),
            "source_train_path": _relative_to_run(train_path, final_run_dir),
            "source_test_path_not_read": _relative_to_run(
                source["test_path"], final_run_dir
            ),
            "source_train_rows": train_rows,
            "source_train_label_counts": expected_label_counts,
            "validation_method": "GroupShuffleSplit",
            "validation_fraction_target": VALIDATION_FRACTION,
            "random_state": VALIDATION_RANDOM_STATE,
            "group_definition": "uint64 hash of all 78 numeric features",
            "hash_key": VALIDATION_HASH_KEY,
            "feature_columns": feature_columns,
            "metadata_columns": metadata_columns,
            "partition_rows": streamed_rows,
            "partition_label_counts": {
                partition: {
                    label: int(streamed_label_counts[partition][code])
                    for code, label in enumerate(LABELS)
                }
                for partition in ("TrainInternal", "Validation")
            },
            "unique_feature_groups": {
                "TrainInternal": int(len(internal_group_ids)),
                "Validation": int(len(validation_group_ids)),
            },
            "shared_feature_groups": int(shared_groups.size),
            "train_internal_path": train_internal_final.name,
            "validation_path": validation_final.name,
            "test_read_or_used": False,
            "model_fitted": False,
        }
        if float_precision is not None:
            manifest["split_csv_float_precision"] = float_precision
        with (temp_run_dir / "validation_manifest.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

        for path in (internal_temp_path, validation_temp_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"File Validation output rỗng hoặc thiếu: {path}")
            output_header = pd.read_csv(path, compression="gzip", nrows=0)
            if output_header.columns.tolist() != expected_columns:
                raise AssertionError(f"Schema output không khớp: {path.name}")
        os.replace(temp_run_dir, final_run_dir)
    finally:
        if temp_run_dir.exists():
            shutil.rmtree(temp_run_dir)

    return {
        "run_dir": final_run_dir,
        "manifest": manifest,
        "partition_audit": partition_audit,
        "day_label_audit": day_label_audit,
        "original_label_audit": original_label_audit,
    }
