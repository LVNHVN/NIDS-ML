from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import GroupShuffleSplit

from validation_split import EXPECTED_METADATA, LABELS, VALIDATION_HASH_KEY, _validate_chunk


TRAIN_DAYS = ("Monday", "Tuesday", "Wednesday")
TEST_DAYS = ("Thursday", "Friday")
DAYS = TRAIN_DAYS + TEST_DAYS
CHUNK_ROWS = 20_000


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def _signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _check_signature(path: Path, expected: dict) -> None:
    if _signature(path) != expected:
        raise RuntimeError(f"File dữ liệu đã thay đổi trong quá trình xử lý: {path}")


def preflight(dataset_path: Path) -> dict:
    dataset_path = Path(dataset_path).resolve()
    columns = pd.read_csv(dataset_path, compression="gzip", nrows=0).columns.tolist()
    features = [column for column in columns if column not in EXPECTED_METADATA]
    if (
        len(columns) != 82
        or len(features) != 78
        or len(set(columns)) != len(columns)
        or any(column not in columns for column in EXPECTED_METADATA)
    ):
        raise ValueError("Dataset phải có 78 feature và bốn metadata duy nhất.")
    dtypes = {**{column: np.float64 for column in features},
              **{column: str for column in EXPECTED_METADATA}}
    sample = pd.read_csv(dataset_path, compression="gzip", nrows=5_000, dtype=dtypes)
    if sample.empty or sample.columns.tolist() != columns:
        raise ValueError("Dataset rỗng hoặc header không khớp.")
    bytes_per_row = sample.memory_usage(index=True, deep=True).sum() / len(sample)
    available = psutil.virtual_memory().available
    chunk_rows = min(CHUNK_ROWS, int(available * 0.10 / max(bytes_per_row * 3, 1)))
    if chunk_rows < 1_000:
        raise MemoryError("RAM khả dụng không đủ cho chunk tối thiểu 1.000 dòng.")
    free = shutil.disk_usage(dataset_path.parent).free
    disk_estimate = max(dataset_path.stat().st_size * 3, 512 * 1024**2)
    if free < disk_estimate:
        raise OSError("Dung lượng đĩa không đủ cho audit và split P2 ước tính.")
    print(f"RAM khả dụng: {available / 1024**3:.2f} GiB; chunk: {chunk_rows:,} dòng.")
    print(f"Đĩa trống: {free / 1024**3:.2f} GiB; ước tính: {disk_estimate / 1024**3:.2f} GiB.")
    print("Ước tính tài nguyên không bảo đảm mức sử dụng thực tế.")
    return {"columns": columns, "features": features, "chunk_rows": chunk_rows}


def _chunks(path: Path, context: dict, *, round_trip: bool = False):
    dtypes = {**{column: np.float64 for column in context["features"]},
              **{column: str for column in EXPECTED_METADATA}}
    yield from pd.read_csv(
        path, compression="gzip", chunksize=context["chunk_rows"], dtype=dtypes,
        float_precision="round_trip" if round_trip else None,
    )


def _validated(chunk: pd.DataFrame, context: dict) -> np.ndarray:
    hashes, _ = _validate_chunk(
        chunk, context["columns"], context["features"], EXPECTED_METADATA
    )
    days = chunk["CaptureDay"]
    if not days.isin(DAYS).all():
        raise ValueError("CaptureDay nằm ngoài Monday–Friday.")
    source_days = chunk["SourceFile"].str.lower()
    if not all(
        source_days[days.eq(day)].str.contains(day.lower(), regex=False).all()
        for day in DAYS
    ):
        raise ValueError("CaptureDay không khớp ngày trong SourceFile.")
    return hashes


def _add_hashes(db: sqlite3.Connection, hashes: np.ndarray, bit: int) -> None:
    db.executemany(
        "INSERT INTO feature_groups(feature_hash, partition_mask) VALUES (?, ?) "
        "ON CONFLICT(feature_hash) DO UPDATE SET "
        "partition_mask = feature_groups.partition_mask | excluded.partition_mask",
        ((f"{int(value):016x}", bit) for value in np.unique(hashes)),
    )


def _new_hash_table(db: sqlite3.Connection) -> None:
    db.execute(
        "CREATE TABLE feature_groups "
        "(feature_hash TEXT PRIMARY KEY, partition_mask INTEGER NOT NULL)"
    )


def _overlap_hashes(db: sqlite3.Connection) -> np.ndarray:
    return np.fromiter(
        (int(row[0], 16) for row in db.execute(
            "SELECT feature_hash FROM feature_groups WHERE partition_mask = 3"
        )),
        dtype=np.uint64,
    )


def audit_dataset(dataset_path: Path, run_dir: Path, context: dict) -> dict:
    dataset_path, run_dir = Path(dataset_path).resolve(), Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run P2 đã tồn tại; không ghi đè: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    signature = _signature(dataset_path)
    counts = Counter()
    day_counts = Counter()
    with closing(sqlite3.connect(run_dir / "feature_groups.sqlite")) as db:
        _new_hash_table(db)
        for chunk in _chunks(dataset_path, context):
            hashes = _validated(chunk, context)
            for (day, label), count in chunk.groupby(
                ["CaptureDay", "BinaryLabel"]
            ).size().items():
                partition = "Train" if day in TRAIN_DAYS else "Test"
                counts[(partition, label)] += int(count)
                day_counts[(day, label)] += int(count)
            for bit, days in ((1, TRAIN_DAYS), (2, TEST_DAYS)):
                mask = chunk["CaptureDay"].isin(days).to_numpy()
                _add_hashes(db, hashes[mask], bit)
        db.commit()
        overlap = _overlap_hashes(db)
        group_counts = {
            "Train": db.execute(
                "SELECT COUNT(*) FROM feature_groups WHERE (partition_mask & 1) != 0"
            ).fetchone()[0],
            "Test": db.execute(
                "SELECT COUNT(*) FROM feature_groups WHERE (partition_mask & 2) != 0"
            ).fetchone()[0],
            "Shared": len(overlap),
        }
    _check_signature(dataset_path, signature)
    if set(day for day, _ in day_counts) != set(DAYS):
        raise ValueError("Dataset không có đủ năm ngày của protocol P2.")
    if any(counts[(partition, label)] == 0 for partition in ("Train", "Test") for label in LABELS):
        raise ValueError("Train và Test đều cần có Normal và Attack.")

    overlaps = Counter()
    for chunk in _chunks(dataset_path, context):
        hashes = _validated(chunk, context)
        mask = np.isin(hashes, overlap)
        for (day, source, label), count in chunk.loc[mask].groupby(
            ["CaptureDay", "SourceFile", "BinaryLabel"]
        ).size().items():
            overlaps[("Train" if day in TRAIN_DAYS else "Test", day, source, label)] += int(count)
    _check_signature(dataset_path, signature)
    day_audit = pd.DataFrame(
        [{"CaptureDay": day, "BinaryLabel": label, "Rows": day_counts[(day, label)]}
         for day in DAYS for label in LABELS]
    )
    partition_audit = pd.DataFrame(
        [{"Partition": part, "Normal": counts[(part, "Normal")],
          "Attack": counts[(part, "Attack")],
          "Rows": sum(counts[(part, label)] for label in LABELS),
          "UniqueFeatureGroups": group_counts[part]}
         for part in ("Train", "Test")]
    )
    overlap_audit = pd.DataFrame(
        [{"Partition": part, "CaptureDay": day, "SourceFile": source,
          "BinaryLabel": label, "Rows": count}
         for (part, day, source, label), count in sorted(overlaps.items())],
        columns=["Partition", "CaptureDay", "SourceFile", "BinaryLabel", "Rows"],
    )
    day_audit.to_csv(run_dir / "day_label_audit.csv", index=False, encoding="utf-8")
    partition_audit.to_csv(run_dir / "partition_audit.csv", index=False, encoding="utf-8")
    overlap_audit.to_csv(run_dir / "overlap_rows_audit.csv", index=False, encoding="utf-8")
    manifest = {
        "run_id": run_dir.name, "source_dataset": os.path.relpath(dataset_path, run_dir),
        "source_signature": signature, "columns": context["columns"],
        "feature_columns": context["features"], "chunk_rows": context["chunk_rows"],
        "train_days": list(TRAIN_DAYS), "test_days": list(TEST_DAYS),
        "counts": {part: {label: counts[(part, label)] for label in LABELS}
                   for part in ("Train", "Test")},
        "feature_groups": group_counts, "hash_key": VALIDATION_HASH_KEY,
        "training_started": False,
    }
    _write_json(run_dir / "audit_manifest.json", manifest)
    print(f"Nhóm hash có mặt ở cả hai phía: {len(overlap):,}.")
    return {"day_audit": day_audit, "partition_audit": partition_audit,
            "overlap_audit": overlap_audit}


def _audit_context(run_dir: Path) -> tuple[dict, Path, dict]:
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir / "audit_manifest.json")
    if (
        manifest["run_id"] != run_dir.name
        or manifest["train_days"] != list(TRAIN_DAYS)
        or manifest["test_days"] != list(TEST_DAYS)
        or manifest["hash_key"] != VALIDATION_HASH_KEY
        or manifest["training_started"] is not False
    ):
        raise ValueError("Audit manifest không khớp protocol P2.")
    path = (run_dir / manifest["source_dataset"]).resolve()
    _check_signature(path, manifest["source_signature"])
    context = {"columns": manifest["columns"], "features": manifest["feature_columns"],
               "chunk_rows": manifest["chunk_rows"]}
    return manifest, path, context


def create_split(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    manifest, dataset_path, context = _audit_context(run_dir)
    outputs = [run_dir / name for name in (
        "p2_train.csv.gz", "p2_test.csv.gz", "split_manifest.json",
        "removed_train_audit.csv",
    )]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Artifact split P2 đã tồn tại; không ghi đè.")
    with closing(sqlite3.connect(run_dir / "feature_groups.sqlite")) as db:
        overlap = _overlap_hashes(db)
    if len(overlap) != manifest["feature_groups"]["Shared"]:
        raise ValueError("SQLite overlap không khớp audit manifest.")
    removed = Counter()
    counts = {part: Counter() for part in ("Train", "Test")}
    source_digests = {part: hashlib.sha256() for part in counts}
    with tempfile.TemporaryDirectory(prefix=".p2_split_", dir=run_dir) as temp:
        temp_dir = Path(temp)
        paths = {part: temp_dir / f"{part}.csv.gz" for part in ("Train", "Test")}
        with gzip.open(paths["Train"], "wt", encoding="utf-8", newline="") as train_file, \
             gzip.open(paths["Test"], "wt", encoding="utf-8", newline="") as test_file:
            files = {"Train": train_file, "Test": test_file}
            headers = {"Train": True, "Test": True}
            for chunk in _chunks(dataset_path, context):
                hashes = _validated(chunk, context)
                is_train = chunk["CaptureDay"].isin(TRAIN_DAYS).to_numpy()
                is_overlap = np.isin(hashes, overlap)
                for (day, source, label), count in chunk.loc[
                    is_train & is_overlap
                ].groupby(["CaptureDay", "SourceFile", "BinaryLabel"]).size().items():
                    removed[(day, source, label)] += int(count)
                for part, mask in (("Train", is_train & ~is_overlap), ("Test", ~is_train)):
                    frame = chunk.loc[mask]
                    if not frame.empty:
                        frame.to_csv(files[part], index=False, header=headers[part])
                        headers[part] = False
                        counts[part].update(frame["BinaryLabel"].value_counts().to_dict())
                        source_digests[part].update(
                            pd.util.hash_pandas_object(frame, index=False).to_numpy(
                                dtype=np.uint64, copy=False
                            ).tobytes()
                        )
            if any(headers.values()):
                raise ValueError("Train hoặc Test sau lọc rỗng.")
        _check_signature(dataset_path, manifest["source_signature"])
        overlap_audit = pd.read_csv(run_dir / "overlap_rows_audit.csv", dtype=str)
        audited_removed = {
            (row.CaptureDay, row.SourceFile, row.BinaryLabel): int(row.Rows)
            for row in overlap_audit.itertuples(index=False) if row.Partition == "Train"
        }
        if dict(removed) != audited_removed:
            raise AssertionError("Dòng Train bị loại không khớp overlap audit.")
        for label in LABELS:
            if counts["Test"][label] != manifest["counts"]["Test"][label]:
                raise AssertionError("Test không bảo toàn số dòng theo nhãn.")
            if counts["Train"][label] + sum(
                count for (_, _, name), count in removed.items() if name == label
            ) != manifest["counts"]["Train"][label]:
                raise AssertionError("Train không bảo toàn số dòng theo nhãn.")
            if counts["Train"][label] == 0 or counts["Test"][label] == 0:
                raise ValueError("Train và Test sau lọc cần cả hai nhãn.")

        with closing(sqlite3.connect(temp_dir / "verify.sqlite")) as db:
            _new_hash_table(db)
            for part, bit, days in (("Train", 1, TRAIN_DAYS), ("Test", 2, TEST_DAYS)):
                observed = Counter()
                output_digest = hashlib.sha256()
                for chunk in _chunks(paths[part], context, round_trip=True):
                    hashes = _validated(chunk, context)
                    if not chunk["CaptureDay"].isin(days).all():
                        raise AssertionError("Ngày trong split không khớp partition.")
                    observed.update(chunk["BinaryLabel"].value_counts().to_dict())
                    _add_hashes(db, hashes, bit)
                    output_digest.update(
                        pd.util.hash_pandas_object(chunk, index=False).to_numpy(
                            dtype=np.uint64, copy=False
                        ).tobytes()
                    )
                if observed != counts[part]:
                    raise AssertionError("Split ghi ra không khớp số dòng theo nhãn.")
                if output_digest.digest() != source_digests[part].digest():
                    raise AssertionError("Nội dung split ghi ra không khớp dữ liệu được chọn.")
            db.commit()
            shared = len(_overlap_hashes(db))
        if shared:
            raise AssertionError(f"Split đã ghi còn {shared} nhóm hash giao nhau.")

        removed_df = pd.DataFrame(
            [{"CaptureDay": day, "SourceFile": source, "BinaryLabel": label,
              "RowsRemoved": count} for (day, source, label), count in sorted(removed.items())],
            columns=["CaptureDay", "SourceFile", "BinaryLabel", "RowsRemoved"],
        )
        partition_df = pd.DataFrame(
            [{"Partition": part, "Rows": sum(counts[part].values()),
              "Normal": counts[part]["Normal"], "Attack": counts[part]["Attack"]}
             for part in ("Train", "Test")]
        )
        removed_df.to_csv(temp_dir / "removed_train_audit.csv", index=False, encoding="utf-8")
        split_manifest = {
            "run_id": run_dir.name, "split_method": "Day-Disjoint",
            "overlap_policy": "keep_all_test_and_remove_train_rows_whose_feature_hash_occurs_in_test",
            "test_features_used_to_filter_train": True,
            "train_path": "p2_train.csv.gz", "test_path": "p2_test.csv.gz",
            "feature_columns": context["features"], "columns": context["columns"],
            "metadata_columns": EXPECTED_METADATA, "split_csv_float_precision": "round_trip",
            "counts": {part: {label: counts[part][label] for label in LABELS}
                       for part in ("Train", "Test")},
            "removed_train_rows": sum(removed.values()),
            "shared_feature_groups_after_filtering": shared,
            "training_started": False,
        }
        _write_json(temp_dir / "split_manifest.json", split_manifest)
        for source, target in (
            (paths["Train"], outputs[0]), (paths["Test"], outputs[1]),
            (temp_dir / "removed_train_audit.csv", outputs[3]),
            (temp_dir / "split_manifest.json", outputs[2]),
        ):
            os.rename(source, target)
    return {"partition_audit": partition_df, "removed_audit": removed_df,
            "manifest": split_manifest}


def create_validation_split(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir / "split_manifest.json")
    if (
        manifest["run_id"] != run_dir.name
        or manifest["split_method"] != "Day-Disjoint"
        or manifest["overlap_policy"] != "keep_all_test_and_remove_train_rows_whose_feature_hash_occurs_in_test"
        or manifest["test_features_used_to_filter_train"] is not True
        or manifest["shared_feature_groups_after_filtering"] != 0
        or manifest["split_csv_float_precision"] != "round_trip"
        or manifest["training_started"] is not False
    ):
        raise ValueError("Split manifest không khớp protocol P2.")
    train_path = run_dir / "p2_train.csv.gz"
    signature = _signature(train_path)
    context = {"columns": manifest["columns"], "features": manifest["feature_columns"],
               "chunk_rows": CHUNK_ROWS}
    expected = manifest["counts"]["Train"]
    rows = sum(expected.values())
    available = psutil.virtual_memory().available
    estimate = rows * 128 + CHUNK_ROWS * 82 * 8 * 3
    if estimate > available * 0.5:
        raise MemoryError("RAM ước tính không đủ cho chia Validation.")
    if shutil.disk_usage(run_dir).free < max(train_path.stat().st_size * 2, 512 * 1024**2):
        raise OSError("Đĩa trống không đủ cho chia Validation.")
    groups, labels = [], Counter()
    for chunk in _chunks(train_path, context, round_trip=True):
        hashes = _validated(chunk, context)
        if not chunk["CaptureDay"].isin(TRAIN_DAYS).all():
            raise ValueError("Train có ngày ngoài Monday–Wednesday.")
        groups.append(hashes)
        labels.update(chunk["BinaryLabel"].value_counts().to_dict())
    _check_signature(train_path, signature)
    if dict(labels) != expected:
        raise AssertionError("Nhãn Train không khớp split manifest.")
    group_ids = np.concatenate(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    internal_indices, validation_indices = next(splitter.split(group_ids, groups=group_ids))
    internal_groups = np.unique(group_ids[internal_indices])
    validation_groups = np.unique(group_ids[validation_indices])
    if np.intersect1d(internal_groups, validation_groups, assume_unique=True).size:
        raise AssertionError("Train nội bộ và Validation có hash giao nhau.")
    expected_rows = {"TrainInternal": len(internal_indices), "Validation": len(validation_indices)}
    validation_root = run_dir / "validation_runs"
    validation_root.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    final_dir = validation_root / run_id
    if final_dir.exists():
        raise FileExistsError(f"Validation run đã tồn tại: {final_dir}")
    with tempfile.TemporaryDirectory(prefix=".validation_", dir=validation_root) as temp:
        temp_dir = Path(temp)
        paths = {"TrainInternal": temp_dir / "train_internal.csv.gz",
                 "Validation": temp_dir / "validation.csv.gz"}
        counts = {part: Counter() for part in paths}
        source_digests = {part: hashlib.sha256() for part in paths}
        day_counts, original_counts = Counter(), Counter()
        with gzip.open(paths["TrainInternal"], "wt", encoding="utf-8", newline="") as internal, \
             gzip.open(paths["Validation"], "wt", encoding="utf-8", newline="") as validation:
            files = {"TrainInternal": internal, "Validation": validation}
            headers = {part: True for part in paths}
            for chunk in _chunks(train_path, context, round_trip=True):
                hashes = _validated(chunk, context)
                mask = np.isin(hashes, validation_groups)
                for part, keep in (("TrainInternal", ~mask), ("Validation", mask)):
                    frame = chunk.loc[keep]
                    if frame.empty:
                        continue
                    frame.to_csv(files[part], index=False, header=headers[part])
                    headers[part] = False
                    counts[part].update(frame["BinaryLabel"].value_counts().to_dict())
                    source_digests[part].update(
                        pd.util.hash_pandas_object(frame, index=False).to_numpy(
                            dtype=np.uint64, copy=False
                        ).tobytes()
                    )
                    for key, count in frame.groupby(["CaptureDay", "BinaryLabel"]).size().items():
                        day_counts[(part, *key)] += int(count)
                    for key, count in frame.groupby(["OriginalLabel", "BinaryLabel"]).size().items():
                        original_counts[(part, *key)] += int(count)
        _check_signature(train_path, signature)
        if any(sum(counts[part].values()) != expected_rows[part] for part in counts):
            raise AssertionError("Số dòng Validation không khớp phân chia nhóm.")
        for label in LABELS:
            if sum(counts[part][label] for part in counts) != expected[label]:
                raise AssertionError("Validation không bảo toàn số dòng theo nhãn.")
            if any(counts[part][label] == 0 for part in counts):
                raise ValueError("Train nội bộ và Validation cần cả hai nhãn.")
        for part in paths:
            observed = Counter()
            output_digest = hashlib.sha256()
            for chunk in _chunks(paths[part], context, round_trip=True):
                hashes = _validated(chunk, context)
                membership = np.isin(hashes, validation_groups)
                if not np.all(membership if part == "Validation" else ~membership):
                    raise AssertionError("Hash ở split Validation đã ghi không đúng partition.")
                observed.update(chunk["BinaryLabel"].value_counts().to_dict())
                output_digest.update(
                    pd.util.hash_pandas_object(chunk, index=False).to_numpy(
                        dtype=np.uint64, copy=False
                    ).tobytes()
                )
            if observed != counts[part]:
                raise AssertionError("Split Validation đã ghi không khớp nhãn.")
            if output_digest.digest() != source_digests[part].digest():
                raise AssertionError("Nội dung Validation đã ghi không khớp Train được chia.")
        partition_df = pd.DataFrame(
            [{"Partition": part, "Rows": sum(counts[part].values()),
              "Normal": counts[part]["Normal"], "Attack": counts[part]["Attack"],
              "AttackRate": counts[part]["Attack"] / sum(counts[part].values()),
              "UniqueFeatureGroups": len(internal_groups if part == "TrainInternal" else validation_groups)}
             for part in paths]
        )
        day_df = pd.DataFrame(
            [{"Partition": part, "CaptureDay": day, "BinaryLabel": label, "Rows": count}
             for (part, day, label), count in sorted(day_counts.items())]
        )
        original_df = pd.DataFrame(
            [{"Partition": part, "OriginalLabel": original, "BinaryLabel": label, "Rows": count}
             for (part, original, label), count in sorted(original_counts.items())]
        )
        partition_df.to_csv(temp_dir / "partition_audit.csv", index=False, encoding="utf-8")
        day_df.to_csv(temp_dir / "day_label_audit.csv", index=False, encoding="utf-8")
        original_df.to_csv(temp_dir / "original_label_audit.csv", index=False, encoding="utf-8")
        _write_json(temp_dir / "validation_manifest.json", {
            "validation_run_id": run_id, "source_run_id": run_dir.name,
            "protocol": "P2", "validation_method": "GroupShuffleSplit",
            "validation_fraction_target": 0.20, "random_state": 42,
            "feature_columns": context["features"], "metadata_columns": EXPECTED_METADATA,
            "partition_rows": {part: sum(counts[part].values()) for part in counts},
            "partition_label_counts": {part: {label: counts[part][label] for label in LABELS}
                                       for part in counts},
            "shared_feature_groups": 0, "split_csv_float_precision": "round_trip",
            "test_read_during_validation": False,
            "test_features_used_to_filter_source_train": True, "model_fitted": False,
        })
        os.rename(temp_dir, final_dir)
    return {"run_dir": final_dir, "partition_audit": partition_df,
            "day_label_audit": day_df, "original_label_audit": original_df}
