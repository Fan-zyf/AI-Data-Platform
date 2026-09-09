"""应用配置。

配置优先从环境变量（或项目根目录 .env 文件）读取，未设置时使用默认值。
"""

import os


class Settings:
    """集中管理后端配置。"""

    # 应用基础信息
    app_name: str = os.getenv("APP_NAME", "AI Data Platform")
    version: str = "0.1.0"

    # 服务监听地址
    host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    port: int = int(os.getenv("BACKEND_PORT", "8000"))

    # CORS 允许来源列表（英文逗号分隔）
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "BACKEND_CORS_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    ]

    # 数据上传限制
    data_max_upload_mb: int = int(os.getenv("DATA_MAX_UPLOAD_MB", "20"))
    data_allowed_extensions: tuple[str, ...] = ("csv", "xlsx", "xls")
    data_preview_rows: int = 20

    @property
    def data_max_upload_bytes(self) -> int:
        """文件大小上限（字节）。"""
        return self.data_max_upload_mb * 1024 * 1024


settings = Settings()
