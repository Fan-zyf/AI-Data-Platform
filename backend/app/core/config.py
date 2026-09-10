"""应用配置。

配置优先从环境变量（或项目根目录 .env 文件）读取，未设置时使用默认值。
"""

import os
from pathlib import Path

# backend 目录绝对路径（backend/app/core/config.py 的上级两级）
_BACKEND_DIR = Path(__file__).resolve().parents[2]


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

    # ---- Dataset Session（临时数据集会话）----
    # Session 有效时长（分钟），默认 2 小时；无后台定时任务，访问/上传时惰性清理
    dataset_session_ttl_minutes: int = int(
        os.getenv("DATASET_SESSION_TTL_MINUTES", "120")
    )
    # 会话数据根目录（默认 backend/runtime/datasets，已被 .gitignore 忽略）
    dataset_runtime_dir: str = os.getenv(
        "DATASET_RUNTIME_DIR", str(_BACKEND_DIR / "runtime" / "datasets")
    )

    # ---- 自动 EDA ----
    eda_max_correlation_columns: int = 30  # 参与相关性矩阵的最大数值字段数
    eda_top_values: int = 10  # 分类变量 Top N
    eda_top_correlations: int = 10  # 相关性 Top N
    eda_max_histogram_bins: int = 40  # 直方图最大分箱数
    eda_min_histogram_bins: int = 6  # 直方图最小分箱数
    eda_strong_correlation_threshold: float = 0.7  # 强相关阈值（|r|）
    eda_high_missing_threshold_percent: float = 20.0  # 高缺失率阈值（%）
    eda_low_cardinality_text_ratio: float = 0.1  # 低基数 text 字段纳入分类分析的唯一值占比上限

    # ---- 机器学习（v0.5：训练 / CV / 评估 / 实验 / 预测）----
    # 单次训练使用的最大行数（超过则拒绝，防止内存/耗时失控）
    ml_max_rows: int = int(os.getenv("ML_MAX_ROWS", "100000"))
    # 训练特征最大个数（自动选择阶段的上限）
    ml_max_features: int = int(os.getenv("ML_MAX_FEATURES", "500"))
    # One-Hot 展开后编码维度上限（防止高基数字段导致特征爆炸）
    ml_max_encoded_features: int = int(os.getenv("ML_MAX_ENCODED_FEATURES", "5000"))
    # 预测接口单次最大记录数
    ml_max_prediction_records: int = int(os.getenv("ML_MAX_PREDICTION_RECORDS", "1000"))
    # 训练所需的最小有效样本数（去掉目标缺失行之后）
    ml_min_total_rows: int = int(os.getenv("ML_MIN_TOTAL_ROWS", "20"))
    # 分类任务允许的最大类别数
    ml_max_unique_classes: int = 50
    # 低基数文本可当分类特征使用的唯一值个数 / 占比上限
    ml_low_cardinality_text_unique: int = 50
    ml_low_cardinality_text_ratio: float = 0.1
    # 高基数标识符（类 ID 数值列）自动排除的唯一值占比阈值
    ml_id_like_unique_ratio: float = 0.95
    # 默认 Task / split / CV 参数
    ml_default_test_size: float = 0.2
    ml_default_random_state: int = 42
    ml_default_cv_folds: int = 5
    # 结果可视化点数上限（ROC 曲线 / 回归散点压缩用）
    ml_roc_max_points: int = 200
    ml_scatter_max_points: int = 500

    # ---- 模型可解释性（v0.6：SHAP）----
    # 解释使用的最大样本数（从源版本随机采样，避免大数据集 SHAP 超时）
    ml_shap_sample_size: int = int(os.getenv("ML_SHAP_SAMPLE_SIZE", "200"))
    # KernelExplainer 的背景摘要样本数（kmeans 摘要）
    ml_shap_background_size: int = int(os.getenv("ML_SHAP_BACKGROUND_SIZE", "50"))
    # SHAP 单点解释（前端展示）的最大条数
    ml_shap_top_features: int = int(os.getenv("ML_SHAP_TOP_FEATURES", "30"))
    # SHAP 计算超时（秒），超过则强制降级或返回错误
    ml_shap_timeout_seconds: int = int(os.getenv("ML_SHAP_TIMEOUT_SECONDS", "120"))


settings = Settings()
