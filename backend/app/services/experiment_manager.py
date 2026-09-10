"""v0.5 Experiment Manager：ML 实验的持久化 / 列表 / 详情 / 删除 / 引用检查。

文件布局（全部位于 Dataset Session 内，随会话删除；runtime 已被 .gitignore 忽略）：

    runtime/datasets/<dataset_id>/
    └── ml/
        └── experiments/
            └── <experiment_id>/          # experiment_id = 服务端 uuid4
                ├── model.joblib          # 完整 Best Pipeline（preprocessing + model）
                ├── metadata.json         # 核心元信息（可追溯 / 可复现）
                └── evaluation.json       # CV 对比 / 最终测试评估结果

设计约定：
- experiment_id 严格 UUID 校验，路径全部由服务端拼接，禁止客户端传入路径；
- 原子写入：先写 .tmp-<uuid> 目录，全部成功后再 rename，失败不留半成品；
- original 派生版本删除保护：若某版本已被 ML 实验作为 source_version_id 引用，
  DELETE version 会返回 409 version_in_use_by_ml_experiment。
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from app.services import dataset_manager
from app.services.dataset_manager import DatasetSessionError

logger = logging.getLogger(__name__)

ML_DIR_NAME = "ml"
EXPERIMENTS_DIR_NAME = "experiments"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
EVALUATION_FILE = "evaluation.json"
# v0.6 SHAP 解释结果（位于每个 experiment 目录下，随 experiment 删除而清理）
SHAP_RESULT_FILE = "shap_result.json"


class ExperimentError(DatasetSessionError):
    """ML 实验管理的业务异常（携带 code / message / HTTP status）。"""


def _experiment_error(code: str, message: str, status_code: int = 400) -> ExperimentError:
    return ExperimentError(code, message, status_code)


def normalize_experiment_id(raw: str) -> str:
    """实验 ID 必须是服务端生成的合法 UUID。"""
    value = (raw or "").strip()
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise _experiment_error(
            "invalid_experiment_id",
            "experiment_id 格式不正确，请使用合法的 UUID。",
            status_code=400,
        ) from None


def experiments_root(session_directory: Path) -> Path:
    return session_directory / ML_DIR_NAME / EXPERIMENTS_DIR_NAME


def _load_session_dir(dataset_id: str) -> Path:
    session = dataset_manager.load_session(dataset_id)
    return session.directory


def _read_json_file(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - 损坏元信息
        logger.warning("cannot read experiment json: %s", path)
        raise _experiment_error(
            "experiment_storage_error",
            "实验元信息读取失败或已损坏。",
            status_code=500,
        ) from exc
    if not isinstance(payload, dict):
        raise _experiment_error(
            "experiment_storage_error",
            "实验元信息结构不完整。",
            status_code=500,
        )
    return payload


def persist_experiment(
    dataset_id: str,
    experiment_id: str,
    pipeline,
    metadata: dict,
    evaluation: dict,
) -> Path:
    """原子写入一个完成的实验（model.joblib + metadata.json + evaluation.json）。

    任何一步失败都会清理临时目录，绝不产生半成品 experiment。
    """
    normalized = normalize_experiment_id(experiment_id)
    session_dir = _load_session_dir(dataset_id)
    root = experiments_root(session_dir)
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / normalized
    tmp_dir = root / f".tmp-{normalized}"
    try:
        tmp_dir.mkdir(parents=False, exist_ok=False)
        # 完整 Pipeline（preprocessing + model）必须整体持久化
        import joblib

        joblib.dump(pipeline, tmp_dir / MODEL_FILE)
        (tmp_dir / METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (tmp_dir / EVALUATION_FILE).write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os_replace_dir(tmp_dir, target_dir)
    except OSError as exc:  # pragma: no cover - 文件系统异常
        logger.exception("failed to persist experiment %s", normalized)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise _experiment_error(
            "experiment_persist_error",
            "保存实验失败，请稍后重试。",
            status_code=500,
        ) from exc
    logger.info("persisted experiment %s (dataset %s)", normalized, dataset_id)
    return target_dir


def os_replace_dir(src: Path, dst: Path) -> None:
    """目录原子替换（os.replace 在 Windows 上要求目标存在或同盘）。

    若目标已存在（理论不发生），先移除后再替换。
    """
    import os

    if dst.exists():  # pragma: no cover - 防御性处理
        shutil.rmtree(dst, ignore_errors=True)
    os.replace(src, dst)


def _iter_experiment_dirs(session_directory: Path) -> list[Path]:
    root = experiments_root(session_directory)
    if not root.is_dir():
        return []
    result: list[Path] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            uuid.UUID(entry.name)
        except ValueError:
            continue
        result.append(entry)
    return result


def _experiment_dir(dataset_id: str, experiment_id: str) -> Path:
    normalized = normalize_experiment_id(experiment_id)
    session_dir = _load_session_dir(dataset_id)
    target = experiments_root(session_dir) / normalized
    if not target.is_dir():
        raise _experiment_error(
            "experiment_not_found",
            "指定的实验不存在或已被删除。",
            status_code=404,
        )
    return target


def list_experiments(dataset_id: str) -> list[dict]:
    """列出全部实验（按 created_at 降序：最新在前）。"""
    session_dir = _load_session_dir(dataset_id)
    items: list[dict] = []
    for directory in _iter_experiment_dirs(session_dir):
        try:
            metadata = _read_json_file(directory / METADATA_FILE)
        except ExperimentError:  # 单个损坏实验跳过，不阻塞整个列表
            logger.warning("skip broken experiment dir: %s", directory)
            continue
        items.append(
            {
                "experiment_id": str(metadata.get("experiment_id", directory.name)),
                "dataset_id": metadata.get("dataset_id"),
                "source_version_id": metadata.get("source_version_id", "original"),
                "created_at": metadata.get("created_at"),
                "target_column": metadata.get("target_column"),
                "task_type": metadata.get("task_type"),
                "best_model": metadata.get("best_model"),
                "primary_metric": metadata.get("primary_metric"),
                "primary_score": metadata.get("best_cv_primary_mean"),
                "test_score_summary": metadata.get("test_score_summary", {}),
                "train_rows": metadata.get("train_rows"),
                "test_rows": metadata.get("test_rows"),
                "effective_cv_folds": metadata.get("effective_cv_folds"),
                "warnings": metadata.get("warnings", []),
                "beats_baseline": metadata.get("beats_baseline"),
            }
        )
    items.sort(key=lambda item: (str(item.get("created_at") or ""), str(item["experiment_id"])), reverse=True)
    return items


def get_experiment(dataset_id: str, experiment_id: str) -> dict:
    """返回实验详情：完整 metadata + evaluation 的合并结果（不含 joblib 二进制）。"""
    target = _experiment_dir(dataset_id, experiment_id)
    metadata = _read_json_file(target / METADATA_FILE)
    evaluation: dict = {}
    if (target / EVALUATION_FILE).is_file():
        evaluation = _read_json_file(target / EVALUATION_FILE)
    combined = dict(metadata)
    combined.update(evaluation)
    combined["experiment_id"] = str(metadata.get("experiment_id", target.name))
    return combined


def load_experiment_model(dataset_id: str, experiment_id: str):
    """加载实验持久化的完整 Pipeline（preprocessing + model），仅供预测阶段调用。

    只允许加载由服务端本模块写入的 model.joblib；目录/ID 均做严格校验。
    """
    target = _experiment_dir(dataset_id, experiment_id)
    model_path = target / MODEL_FILE
    if not model_path.is_file():
        raise _experiment_error(
            "experiment_storage_error",
            "实验模型文件缺失或已被清理，无法执行预测。",
            status_code=500,
        )
    try:
        import joblib

        pipeline = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - 模型反序列化异常
        logger.exception("failed to load model for experiment %s", target.name)
        raise _experiment_error(
            "experiment_model_load_error",
            "加载实验模型失败，模型文件可能已损坏。",
            status_code=500,
        ) from exc
    return pipeline


def delete_experiment(dataset_id: str, experiment_id: str) -> bool:
    """删除实验目录（model.joblib + metadata + evaluation）。"""
    target = _experiment_dir(dataset_id, experiment_id)
    try:
        shutil.rmtree(target)
    except OSError as exc:  # pragma: no cover
        raise _experiment_error(
            "experiment_delete_failed",
            "删除实验失败，请稍后重试。",
            status_code=500,
        ) from exc
    logger.info("deleted experiment %s (dataset %s)", experiment_id, dataset_id)
    return True


def find_experiments_using_version(dataset_id: str, version_id: str) -> list[str]:
    """返回引用了指定 source_version_id 的实验 ID 列表。

    版本删除前调用：存在引用时须返回 409，避免实验元信息指向已删除的数据版本。
    """
    session_dir = _load_session_dir(dataset_id)
    ref = (version_id or "").strip()
    matches: list[str] = []
    for directory in _iter_experiment_dirs(session_dir):
        try:
            metadata = _read_json_file(directory / METADATA_FILE)
        except ExperimentError:
            continue
        if str(metadata.get("source_version_id", "original")) == ref:
            matches.append(str(metadata.get("experiment_id", directory.name)))
    return matches


# ============================================================
# v0.6 模型可解释性：跨会话定位 / SHAP 结果持久化
# ============================================================


def _runtime_root() -> Path:
    """Dataset Session 根目录（runtime/datasets）。"""
    from app.core.config import settings

    return Path(settings.dataset_runtime_dir)


def find_experiment_location(experiment_id: str) -> tuple[str, Path] | None:
    """跨会话扫描，定位 experiment_id 所属的 dataset_id 与目录。

    返回 (dataset_id, experiment_dir)，找不到返回 None。
    仅识别 UUID 格式的目录名，避免误判临时文件。
    """
    normalized = normalize_experiment_id(experiment_id)
    root = _runtime_root()
    if not root.is_dir():
        return None
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        candidate = session_dir / ML_DIR_NAME / EXPERIMENTS_DIR_NAME / normalized
        if candidate.is_dir() and (candidate / METADATA_FILE).is_file():
            return session_dir.name, candidate
    return None


def write_shap_result(
    experiment_dir: Path,
    payload: dict,
    metadata: dict,
) -> Path:
    """把 SHAP 解释结果写入 <experiment>/shap_result.json，并更新 metadata.json。

    payload：完整解释结果（global / summary / samples / 评估信息）。
    metadata：仅解释相关的附加元信息，会 merge 到 metadata.json 的
    ``explainability`` 字段下，供实验详情接口直接读取。
    """
    import os
    import tempfile

    target = Path(experiment_dir) / SHAP_RESULT_FILE
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=".shap_",
            suffix=".tmp",
            delete=False,
        ) as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            tmp_path = Path(fp.name)
        os.replace(tmp_path, target)
    except OSError as exc:  # pragma: no cover - 文件系统异常
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise _experiment_error(
            "shap_persist_error",
            "保存 SHAP 解释结果失败，请稍后重试。",
            status_code=500,
        ) from exc

    # 更新 metadata.json：标记 explainability 状态、记录文件相对路径与时间戳
    meta_path = Path(experiment_dir) / METADATA_FILE
    try:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    existing["explainability"] = {
        "status": "computed",
        "result_path": SHAP_RESULT_FILE,
        "created_at": metadata.get("created_at"),
        "explainer_type": metadata.get("explainer_type"),
        "model_class": metadata.get("model_class"),
        "task_type": metadata.get("task_type"),
        "class_label_used": metadata.get("class_label_used"),
        "n_rows_used": metadata.get("n_rows_used"),
        "warnings": metadata.get("warnings", []),
    }
    meta_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def read_shap_result(experiment_dir: Path) -> dict | None:
    """读取已保存的 SHAP 解释结果；文件不存在或损坏时返回 None。"""
    target = Path(experiment_dir) / SHAP_RESULT_FILE
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - 损坏的 JSON
        return None
