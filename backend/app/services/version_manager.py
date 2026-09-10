"""数据版本管理服务：在 Dataset Session 内维护不可变的派生版本目录。

文件布局（runtime 已被 .gitignore 忽略，绝不提交用户数据到 Git）：

    runtime/datasets/<dataset_id>/
    ├── source.<ext>            # 原始文件（永远保留、不可修改）
    ├── metadata.json           # Dataset Session 元信息（v0.3 结构保持不变）
    └── versions/
        └── <version_id>/       # version_id 为合法 UUID
            ├── data.csv        # 派生版本数据
            └── metadata.json   # 版本元信息

版本模型：original -> version_001 -> version_002 …（单向 parent / child 链）。

设计约定：
- original 永远只读，不落盘为 versions/<uuid>；删除接口明确禁止删除 original；
- version_id 必须校验为 UUID，所有路径均由服务端 UUID + 固定文件名拼接，防止路径穿越；
- 每个派生版本独立目录保存 data.csv + metadata.json，不覆盖任何其他版本；
- 业务异常统一抛出 :class:`VersionError`（继承 DatasetSessionError）。
"""

import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.models.data import QualityOverview
from app.models.processing import ORIGINAL_VERSION
from app.services import data_service as ds
from app.services import dataset_manager
from app.services.data_service import DataServiceError, read_dataframe

logger = logging.getLogger(__name__)

VERSIONS_DIR_NAME = "versions"
VERSION_DATA_FILE = "data.csv"
VERSION_META_FILE = "metadata.json"


class VersionError(dataset_manager.DatasetSessionError):
    """版本管理的业务异常（携带 error code / 友好 message / HTTP status）。"""


def _version_error(code: str, message: str, status_code: int = 400) -> VersionError:
    return VersionError(code, message, status_code)


# ------------------------------------------------------------
# ID 校验与基础路径
# ------------------------------------------------------------


def normalize_version_ref(raw: str | None, allow_original: bool = True) -> str:
    """规范化版本引用：original 原样返回；其他必须为合法 UUID。"""
    value = (raw or "").strip() or ORIGINAL_VERSION
    if value == ORIGINAL_VERSION:
        if not allow_original:
            raise _version_error(
                "original_version_protected",
                "原始版本是只读基线，不允许执行该操作。",
            )
        return ORIGINAL_VERSION
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise _version_error(
            "invalid_version_id",
            "version_id 格式不正确，请使用合法的 UUID。",
        ) from None


def _load_session(dataset_id: str):
    """加载并校验（含 TTL）Dataset Session，同时返回会话目录。"""
    session = dataset_manager.load_session(dataset_id)
    return session, session.directory


def _versions_dir(session_directory: Path) -> Path:
    return session_directory / VERSIONS_DIR_NAME


def _version_dir(session_directory: Path, version_id: str) -> Path:
    return _versions_dir(session_directory) / version_id


def _read_version_meta(version_directory: Path) -> dict:
    meta_path = version_directory / VERSION_META_FILE
    data_path = version_directory / VERSION_DATA_FILE
    if not meta_path.is_file() or not data_path.is_file():
        raise _version_error(
            "version_storage_error",
            "版本数据文件缺失或已损坏，无法读取该版本。",
            status_code=500,
        )
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("invalid version metadata: %s", version_directory)
        raise _version_error(
            "version_storage_error",
            "版本元信息损坏，无法读取该版本。",
            status_code=500,
        ) from exc
    if not isinstance(payload, dict) or "version_id" not in payload:
        raise _version_error(
            "version_storage_error",
            "版本元信息结构不完整，无法读取该版本。",
            status_code=500,
        )
    return payload


def _iter_derived_dirs(session_directory: Path) -> list[Path]:
    """枚举所有派生版本目录（仅合法 UUID 目录）。"""
    vdir = _versions_dir(session_directory)
    if not vdir.is_dir():
        return []
    result: list[Path] = []
    for entry in vdir.iterdir():
        if not entry.is_dir():
            continue
        try:
            uuid.UUID(entry.name)
        except ValueError:
            continue
        result.append(entry)
    return result


# ------------------------------------------------------------
# DataFrame 加载（original / 派生版本共用）
# ------------------------------------------------------------


def _read_csv_content(path: Path) -> pd.DataFrame:
    try:
        content = path.read_bytes()
    except OSError as exc:  # pragma: no cover - 文件系统异常
        raise _version_error(
            "version_storage_error",
            "版本数据文件读取失败。",
            status_code=500,
        ) from exc
    try:
        return read_dataframe("csv", content)
    except DataServiceError as exc:
        raise _version_error(exc.code, exc.message, exc.status_code) from exc


def load_dataset_version(dataset_id: str, version_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    """按版本引用加载 DataFrame，返回 (df, 版本信息 dict)。缺省 version_id 时为 original。

    返回的版本信息包含：version_id / parent_version_id / is_original / created_at /
    rows / columns / file_name；quality 不在此计算，需要时另行调用。
    """
    session, session_dir = _load_session(dataset_id)
    ref = normalize_version_ref(version_id)

    if ref == ORIGINAL_VERSION:
        if not (session_dir / session.source_file_name).is_file():
            raise _version_error(
                "dataset_read_error",
                "数据集源文件缺失，无法读取原始版本。",
                status_code=500,
            )
        try:
            content = (session_dir / session.source_file_name).read_bytes()
            frame = read_dataframe(session.file_type, content)
        except OSError as exc:  # pragma: no cover
            raise _version_error(
                "dataset_read_error", "数据集源文件读取失败。", status_code=500
            ) from exc
        except DataServiceError as exc:
            raise _version_error(exc.code, exc.message, exc.status_code) from exc
        if len(frame) == 0:
            raise _version_error(
                "no_rows", "原始版本数据为空，无法分析。", status_code=422
            )
        info = {
            "version_id": ORIGINAL_VERSION,
            "parent_version_id": None,
            "source_version": ORIGINAL_VERSION,
            "is_original": True,
            "created_at": session.created_at,
            "rows": len(frame),
            "columns": len(frame.columns),
            "file_name": session.original_filename,
            "file_type": session.file_type,
            "file_size": session.file_size,
        }
        return frame, info

    version_dir = _version_dir(session_dir, ref)
    if not version_dir.is_dir():
        raise _version_error(
            "version_not_found",
            "指定的数据版本不存在或已被删除。",
            status_code=404,
        )
    meta = _read_version_meta(version_dir)
    frame = _read_csv_content(version_dir / VERSION_DATA_FILE)
    if len(frame) == 0:
        raise _version_error(
            "version_storage_error",
            "版本数据为空，无法分析。",
            status_code=500,
        )
    created_at_raw = meta.get("created_at")
    info = {
        "version_id": ref,
        "parent_version_id": meta.get("parent_version_id"),
        "source_version": meta.get("source_version", ORIGINAL_VERSION),
        "is_original": False,
        "created_at": _parse_iso(created_at_raw),
        "rows": len(frame),
        "columns": len(frame.columns),
        "file_name": session.original_filename,
        "file_type": "csv",
        "file_size": 0,
        "quality": meta.get("quality"),
        "operations": meta.get("operations", []),
        "column_names": meta.get("column_names", [str(c) for c in frame.columns]),
        "metadata": meta.get("metadata", {}),
    }
    return frame, info


def _parse_iso(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:  # pragma: no cover - 兼容损坏元信息
        return None


# ------------------------------------------------------------
# 派生版本持久化（原子写入）
# ------------------------------------------------------------


def persist_new_version(
    dataset_id: str,
    parent_version_id: str,
    frame: pd.DataFrame,
    operations: list[dict],
    quality: QualityOverview,
    metadata_payload: dict | None = None,
    file_name: str = "derived.csv",
) -> dict:
    """把一次成功的 Transform Apply 结果保存为新的派生版本。

    原子性：先写入 <versions>/.tmp-<uuid> 目录，成功后 rename 为正式目录；
    任何一步失败都会清理临时目录，绝不留下“半成品版本”。
    """
    session, session_dir = _load_session(dataset_id)
    versions_root = _versions_dir(session_dir)
    versions_root.mkdir(parents=True, exist_ok=True)

    version_id = str(uuid.uuid4())
    created_at = datetime.now().astimezone().isoformat(timespec="microseconds")

    column_names = [str(col) for col in frame.columns]
    meta = {
        "version_id": version_id,
        "dataset_id": dataset_id,
        "parent_version_id": normalize_version_ref(parent_version_id),
        "source_version": normalize_version_ref(parent_version_id),
        "file_name": file_name,
        "created_at": created_at,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "operations": operations,
        "column_names": column_names,
        "quality": quality.model_dump(),
        "metadata": metadata_payload or {},
    }

    target_dir = _version_dir(session_dir, version_id)
    tmp_dir = versions_root / f".tmp-{version_id}"
    try:
        tmp_dir.mkdir(parents=False, exist_ok=False)
        (tmp_dir / VERSION_DATA_FILE).write_text(
            frame.to_csv(index=False), encoding="utf-8"
        )
        (tmp_dir / VERSION_META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 目标目录理论不存在；rename 提供原子性
        os.replace(tmp_dir, target_dir)
    except OSError as exc:  # pragma: no cover - 文件系统异常
        logger.exception("failed to persist derived version %s", version_id)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise _version_error(
            "version_write_error",
            "保存新数据版本失败，请稍后重试。",
            status_code=500,
        ) from exc
    logger.info("created derived version %s (parent=%s)", version_id, parent_version_id)
    return meta


# ------------------------------------------------------------
# 版本列表 / 详情 / 对比
# ------------------------------------------------------------


def _metrics_for_frame(frame: pd.DataFrame) -> tuple[dict, QualityOverview]:
    """计算 (列名列表/简单统计, 完整质量概览)。"""
    overview, _ = ds.quality_overview(frame)
    metrics = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing": overview.total_missing,
        "duplicates": overview.duplicate_rows,
    }
    return metrics, overview


def _item_from_meta(meta: dict) -> dict:
    created_at = _parse_iso(meta.get("created_at"))
    quality = meta.get("quality") or {}
    return {
        "version_id": str(meta["version_id"]),
        "parent_version_id": meta.get("parent_version_id"),
        "is_original": False,
        "created_at": created_at,
        "rows": int(meta.get("rows", 0)),
        "columns": int(meta.get("columns", 0)),
        "missing": int(quality.get("total_missing", 0)),
        "duplicates": int(quality.get("duplicate_rows", 0)),
        "operation_count": len(meta.get("operations", [])),
        "operations_summary": _operations_summary(meta.get("operations", [])),
    }


def _original_item(dataset_id: str) -> dict:
    """original 版本列表项（rows/columns 来自 Session，缺失/重复实时统计）。"""
    session, _ = _load_session(dataset_id)
    frame = dataset_manager.load_dataframe(dataset_id)
    metrics, _ = _metrics_for_frame(frame)
    return {
        "version_id": ORIGINAL_VERSION,
        "parent_version_id": None,
        "is_original": True,
        "created_at": session.created_at,
        "rows": metrics["rows"],
        "columns": metrics["columns"],
        "missing": metrics["missing"],
        "duplicates": metrics["duplicates"],
        "operation_count": 0,
        "operations_summary": [],
    }


def _operations_summary(operations: list[dict]) -> list[str]:
    """把操作历史压缩成简短的人类可读摘要，如「1. 缺失值填充(median) → 3列受影响」。"""
    labels = {
        "drop_duplicates": "删除重复行",
        "fill_missing": "缺失值填充",
        "drop_columns": "删除字段",
        "convert_type": "类型转换",
        "remove_outliers": "IQR 异常值处理",
        "text_transform": "文本处理",
        "date_features": "日期特征工程",
        "one_hot_encode": "One-Hot 编码",
        "scale_numeric": "数值缩放",
    }
    summaries: list[str] = []
    for index, op in enumerate(operations, start=1):
        op_type = str(op.get("type", "unknown"))
        label = labels.get(op_type, op_type)
        detail = ""
        if op_type == "fill_missing":
            detail = f"({op.get('strategy', '')})"
        elif op_type in ("drop_columns", "one_hot_encode", "text_transform", "scale_numeric"):
            detail = f"({','.join(op.get('columns', []) or [])[:60]})"
        elif op_type in ("convert_type", "remove_outliers", "date_features"):
            detail = f"({op.get('column', '')})"
        summaries.append(f"{index}. {label}{detail}")
    return summaries


def list_versions(dataset_id: str) -> tuple[dict, list[dict]]:
    """返回 (original 项 dict, 派生版本项列表按创建时间升序)。"""
    original = _original_item(dataset_id)
    _, session_dir = _load_session(dataset_id)
    items: list[dict] = []
    for version_dir in _iter_derived_dirs(session_dir):
        try:
            meta = _read_version_meta(version_dir)
            items.append(_item_from_meta(meta))
        except VersionError:  # 单个版本损坏时跳过，不阻塞整份列表
            logger.warning("skip broken version dir: %s", version_dir)
    items.sort(key=lambda item: (item["created_at"] or datetime.min, item["version_id"]))
    return original, items


def _load_one_for_metrics(dataset_id: str, version_id: str) -> tuple[pd.DataFrame, dict]:
    return load_dataset_version(dataset_id, version_id)


def compare_versions(dataset_id: str, version_a: str, version_b: str) -> dict:
    """A/B 版本基础对比：行列/缺失/重复 + 列集合差异（A→B 方向）。"""
    frame_a, info_a = _load_one_for_metrics(dataset_id, version_a)
    frame_b, info_b = _load_one_for_metrics(dataset_id, version_b)

    metrics_a, _ = _metrics_for_frame(frame_a)
    metrics_b, _ = _metrics_for_frame(frame_b)

    columns_a = set(str(c) for c in frame_a.columns)
    columns_b = set(str(c) for c in frame_b.columns)
    generated = sorted(columns_b - columns_a)
    removed = sorted(columns_a - columns_b)

    return {
        "success": True,
        "dataset_id": dataset_id,
        "version_a": info_a["version_id"],
        "version_b": info_b["version_id"],
        "rows_a": metrics_a["rows"],
        "columns_a": metrics_a["columns"],
        "missing_a": metrics_a["missing"],
        "duplicates_a": metrics_a["duplicates"],
        "rows_b": metrics_b["rows"],
        "columns_b": metrics_b["columns"],
        "missing_b": metrics_b["missing"],
        "duplicates_b": metrics_b["duplicates"],
        "generated_columns": generated,
        "removed_columns": removed,
    }


def version_detail(dataset_id: str, version_id: str) -> dict:
    """版本详情：完整元信息 + 质量 + 列画像 + 前 20 行预览 + 操作历史。"""
    session, _ = _load_session(dataset_id)
    ref = normalize_version_ref(version_id)
    frame, info = load_dataset_version(dataset_id, ref)

    if info["is_original"]:
        quality, _ = ds.quality_overview(frame)
        operations: list[dict] = []
        created_at = session.created_at
    else:
        quality = QualityOverview(**info.get("quality") or {})
        operations = info.get("operations") or []
        created_at = info["created_at"]

    profiles, _ = ds.analyze_columns(frame)
    return {
        "success": True,
        "dataset_id": dataset_id,
        "version_id": ref,
        "parent_version_id": info.get("parent_version_id"),
        "is_original": info["is_original"],
        "created_at": created_at,
        "rows": info["rows"],
        "columns": info["columns"],
        "file_name": info["file_name"],
        "operations": operations,
        "quality": quality,
        "column_profiles": profiles,
        "preview": ds.build_preview_rows(frame),
    }


# ------------------------------------------------------------
# 删除派生版本
# ------------------------------------------------------------


def delete_derived_version(dataset_id: str, version_id: str) -> bool:
    """删除派生版本；original 永远禁止删除；有子版本时返回 409 冲突。"""
    ref = normalize_version_ref(version_id, allow_original=False)
    _, session_dir = _load_session(dataset_id)
    version_dir = _version_dir(session_dir, ref)
    if not version_dir.is_dir():
        raise _version_error(
            "version_not_found",
            "指定的数据版本不存在或已被删除。",
            status_code=404,
        )

    # 检查该版本是否被其他版本作为父版本引用
    for other_dir in _iter_derived_dirs(session_dir):
        if other_dir == version_dir:
            continue
        try:
            other_meta = _read_version_meta(other_dir)
        except VersionError:
            continue
        if str(other_meta.get("parent_version_id", "")) == ref:
            raise _version_error(
                "version_has_children",
                "该版本存在子版本，不能删除。请先删除或调整其子版本。",
                status_code=409,
            )

    # v0.5：检查该版本是否被机器学习实验作为数据源引用
    from app.services import experiment_manager as em

    experiment_refs = em.find_experiments_using_version(dataset_id, ref)
    if experiment_refs:
        raise _version_error(
            "version_in_use_by_ml_experiment",
            "该版本已被机器学习实验作为数据源引用，不能删除。"
            "请先删除相关实验（引用实验："
            + "、".join(experiment_refs[:3])
            + (f" 等 {len(experiment_refs)} 个" if len(experiment_refs) > 3 else "")
            + "）后再删除此版本。",
            status_code=409,
        )

    try:
        shutil.rmtree(version_dir)
    except OSError as exc:  # pragma: no cover
        raise _version_error(
            "version_delete_failed",
            "删除数据版本失败，请稍后重试。",
            status_code=500,
        ) from exc
    logger.info("deleted derived version %s (dataset %s)", ref, dataset_id)
    return True
