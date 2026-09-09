"""临时 Dataset Session 管理服务。

职责：
- 上传成功后生成 UUID dataset_id，并把"本次分析所需的临时数据"保存到本地 runtime 目录；
- 按配置的 TTL 惰性判断 Session 是否过期（访问/上传时顺带清理，无后台定时任务）；
- 后续数据清洗、特征工程、机器学习模块将复用 dataset_id 读取同一份会话数据。

设计约定：
- 不使用数据库，文件布局：runtime/datasets/<dataset_id>/source.<ext> + metadata.json；
- 目录名只使用合法 UUID，不覆盖用户原始文件，不信任用户提交的文件名做路径拼接；
- 本模块只负责会话生命周期，pandas 解析复用 data_service.read_dataframe；
- 业务异常统一抛出 :class:`DatasetSessionError`，由 API 层转换为 HTTP 响应。
"""

import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import settings
from app.services.data_service import DataServiceError, read_dataframe

logger = logging.getLogger(__name__)

# 会话目录内文件命名
SOURCE_FILE_TEMPLATE = "source.{file_type}"
METADATA_FILE = "metadata.json"

# 上传时顺带清理的最大过期 Session 数（防御性上限）
_PURGE_LIMIT = 500


class DatasetSessionError(Exception):
    """Dataset Session 业务异常，携带友好错误信息与建议的 HTTP 状态码。"""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _utcnow() -> datetime:
    """当前 UTC 时间（aware）。"""
    return datetime.now(timezone.utc)


def _runtime_root() -> Path:
    """会话根目录（按需创建）。"""
    root = Path(settings.dataset_runtime_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - 文件系统异常
        raise DatasetSessionError(
            "session_write_error",
            "服务器无法创建数据集临时目录，请稍后重试。",
            status_code=500,
        ) from exc
    return root


def _normalize_dataset_id(dataset_id: str) -> str:
    """校验 dataset_id 必须是合法 UUID，防止目录穿越。"""
    raw = (dataset_id or "").strip()
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError):
        raise DatasetSessionError(
            "invalid_dataset_id",
            "dataset_id 格式不正确，请使用合法的 UUID。",
            status_code=400,
        ) from None
    return str(parsed)


@dataclass
class DatasetSession:
    """数据集会话元信息（对应 metadata.json 内容）。"""

    dataset_id: str
    original_filename: str
    file_type: str
    file_size: int
    rows: int
    columns: int
    created_at: datetime
    expires_at: datetime

    @property
    def directory(self) -> Path:
        return Path(settings.dataset_runtime_dir) / self.dataset_id

    @property
    def source_file_name(self) -> str:
        return SOURCE_FILE_TEMPLATE.format(file_type=self.file_type)

    @property
    def source_path(self) -> Path:
        return self.directory / self.source_file_name


def _dump_metadata(session: DatasetSession) -> None:
    """把会话元信息写入 metadata.json。"""
    payload = {
        "dataset_id": session.dataset_id,
        "original_filename": session.original_filename,
        "file_type": session.file_type,
        "file_size": session.file_size,
        "rows": session.rows,
        "columns": session.columns,
        "created_at": session.created_at.isoformat(),
        "expires_at": session.expires_at.isoformat(),
        "source_file": session.source_file_name,
    }
    (session.directory / METADATA_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_metadata(directory: Path) -> DatasetSession:
    """从 metadata.json 读取会话元信息。"""
    meta_path = directory / METADATA_FILE
    if not meta_path.exists():
        raise DatasetSessionError(
            "dataset_storage_error",
            "数据集会话元信息缺失，文件可能被清理，请重新上传。",
            status_code=500,
        )
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return DatasetSession(
            dataset_id=str(payload["dataset_id"]),
            original_filename=str(payload["original_filename"]),
            file_type=str(payload["file_type"]),
            file_size=int(payload["file_size"]),
            rows=int(payload["rows"]),
            columns=int(payload["columns"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            expires_at=datetime.fromisoformat(payload["expires_at"]),
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("invalid metadata in %s: %s", directory, exc)
        raise DatasetSessionError(
            "dataset_storage_error",
            "数据集会话元信息损坏，文件可能被清理，请重新上传。",
            status_code=500,
        ) from exc


def create_session(
    original_filename: str,
    file_type: str,
    content: bytes,
    rows: int,
    columns: int,
) -> DatasetSession:
    """为一次成功上传创建数据集 Session 并落盘临时数据。"""
    purge_expired()  # 上传新数据时顺便清理过期 Session

    root = _runtime_root()
    dataset_id = str(uuid.uuid4())
    directory = root / dataset_id

    try:
        directory.mkdir(parents=False, exist_ok=False)
    except FileExistsError:  # 理论上不可能，UUID 冲突极小概率兜底
        dataset_id = str(uuid.uuid4())
        directory = root / dataset_id
        directory.mkdir(parents=False, exist_ok=False)

    try:
        created_at = _utcnow()
        ttl_minutes = max(int(settings.dataset_session_ttl_minutes), 0)
        expires_at = created_at + timedelta(minutes=ttl_minutes)
        session = DatasetSession(
            dataset_id=dataset_id,
            original_filename=original_filename,
            file_type=file_type,
            file_size=len(content),
            rows=rows,
            columns=columns,
            created_at=created_at,
            expires_at=expires_at,
        )
        (directory / session.source_file_name).write_bytes(content)
        _dump_metadata(session)
        return session
    except DatasetSessionError:
        raise
    except OSError as exc:  # pragma: no cover - 文件系统异常
        logger.exception("failed to persist dataset session %s", dataset_id)
        shutil.rmtree(directory, ignore_errors=True)
        raise DatasetSessionError(
            "session_write_error",
            "保存数据集会话失败，请稍后重试。",
            status_code=500,
        ) from exc


def load_session(dataset_id: str) -> DatasetSession:
    """加载未过期的会话元信息；不存在返回 404，已过期自动清理并提示。"""
    dsid = _normalize_dataset_id(dataset_id)
    directory = _runtime_root() / dsid

    if not directory.is_dir():
        raise DatasetSessionError(
            "dataset_not_found",
            "数据集不存在或已被删除，请重新上传文件。",
            status_code=404,
        )

    session = _load_metadata(directory)
    if _utcnow() > session.expires_at:
        shutil.rmtree(directory, ignore_errors=True)
        raise DatasetSessionError(
            "dataset_expired",
            "数据集会话已过期，请重新上传文件后再进行分析。",
            status_code=404,
        )
    return session


def load_dataframe(dataset_id: str):
    """读取会话对应的 DataFrame（每次 EDA 只解析一次，供向量化计算使用）。"""
    session = load_session(dataset_id)
    if not session.source_path.is_file():
        raise DatasetSessionError(
            "dataset_read_error",
            "数据集源文件缺失，无法分析，请重新上传文件。",
            status_code=500,
        )
    try:
        content = session.source_path.read_bytes()
    except OSError as exc:  # pragma: no cover - 文件系统异常
        raise DatasetSessionError(
            "dataset_read_error",
            "数据集源文件读取失败，请重新上传文件。",
            status_code=500,
        ) from exc

    try:
        frame = read_dataframe(session.file_type, content)
    except DataServiceError as exc:
        raise DatasetSessionError(exc.code, exc.message, exc.status_code) from exc
    except Exception as exc:  # pragma: no cover - pandas 底层异常
        logger.exception("failed to parse session %s", dataset_id)
        raise DatasetSessionError(
            "dataset_read_error",
            "数据集源文件解析失败，请重新上传文件。",
            status_code=422,
        ) from exc

    if len(frame) == 0:
        raise DatasetSessionError(
            "no_rows",
            "会话数据为空，无法进行 EDA，请重新上传文件。",
            status_code=422,
        )
    return frame


def delete_session(dataset_id: str) -> bool:
    """删除会话目录。不存在时抛 404，删除成功后返回 True。"""
    dsid = _normalize_dataset_id(dataset_id)
    directory = _runtime_root() / dsid
    if not directory.is_dir():
        raise DatasetSessionError(
            "dataset_not_found",
            "数据集不存在或已被删除，无需重复清理。",
            status_code=404,
        )
    try:
        shutil.rmtree(directory, ignore_errors=False)
    except OSError as exc:  # pragma: no cover - 文件系统异常
        raise DatasetSessionError(
            "session_delete_error",
            "数据集会话清理失败，请稍后重试。",
            status_code=500,
        ) from exc
    return True


def purge_expired() -> int:
    """清理所有过期会话目录（无后台定时任务时的惰性清理方案）。"""
    root = Path(settings.dataset_runtime_dir)
    if not root.is_dir():
        return 0

    cleaned = 0
    now = _utcnow()
    for entry in root.iterdir():
        if cleaned >= _PURGE_LIMIT:
            break
        if not entry.is_dir():
            continue
        try:
            uuid.UUID(entry.name)  # 仅处理 UUID 目录，避免误删其他内容
        except ValueError:
            continue
        try:
            meta = _load_metadata(entry)
        except DatasetSessionError:
            continue
        if now > meta.expires_at:
            shutil.rmtree(entry, ignore_errors=True)
            cleaned += 1
            logger.info("purged expired dataset session %s", meta.dataset_id)
    return cleaned
