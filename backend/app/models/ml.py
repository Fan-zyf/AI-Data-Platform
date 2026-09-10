"""v0.5 机器学习模块：训练 / 预测请求与实验列表项 Pydantic 模型。

训练与详情接口的响应体是动态 JSON（classification 与 regression 携带不同的评估结构），
因此不由本文件声明重量级响应模型，统一由 ml_service 生成完全 JSON-safe 的 dict。
错误响应保持项目统一的 {success:false, error:{code,message}}。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---- 任务类型 ----
TaskType = Literal["auto", "classification", "regression"]

# ---- 候选模型注册表 key ----
CLASSIFICATION_MODELS = ("dummy", "logistic_regression", "random_forest")
REGRESSION_MODELS = ("dummy", "ridge", "random_forest")
ALL_MODEL_KEYS = tuple(sorted(set(CLASSIFICATION_MODELS) | set(REGRESSION_MODELS)))

# ---- 展示名映射（前端 / 实验列表共用） ----
MODEL_DISPLAY_NAMES = {
    "dummy": "Dummy Baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "ridge": "Ridge",
}


class MLTrainRequest(BaseModel):
    """POST /ml/train 的训练请求。

    所有随机操作（split / CV / RandomForest …）都使用同一 random_state，
    保证同一请求可复现。
    """

    source_version_id: str | None = Field(
        None, description="数据版本引用；缺省为 original，也可以是派生版本 UUID"
    )
    target_column: str = Field(..., min_length=1, description="预测目标列")
    task_type: TaskType = Field("auto", description="auto / classification / regression")
    feature_columns: list[str] | None = Field(
        None, description="显式特征列表；缺省时后端自动选择"
    )
    exclude_columns: list[str] = Field(default_factory=list, description="额外排除的特征列")
    candidate_models: list[str] | None = Field(
        None, description="候选模型 key；缺省使用当前任务默认集合"
    )
    test_size: float = Field(0.2, description="测试集比例（默认 0.2）")
    random_state: int = Field(42, description="随机种子（默认 42）")
    cv_folds: int = Field(5, description="请求的交叉验证折数（默认 5）")

    @field_validator("test_size")
    @classmethod
    def _test_size_in_range(cls, value: float) -> float:
        if not (0.0 < value < 1.0):
            raise ValueError("test_size 必须在 (0, 1) 区间内")
        return float(value)

    @field_validator("cv_folds")
    @classmethod
    def _cv_folds_in_range(cls, value: int) -> int:
        if value < 2 or value > 20:
            raise ValueError("cv_folds 需在 2~20 之间")
        return int(value)

    @field_validator("random_state")
    @classmethod
    def _random_state_in_range(cls, value: int) -> int:
        if not 0 <= value <= 2**31 - 1:
            raise ValueError("random_state 需为 0 ~ 2^31-1 之间的整数")
        return int(value)

    @field_validator("candidate_models")
    @classmethod
    def _candidate_models_known(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("candidate_models 不能为空数组")
        known = set(ALL_MODEL_KEYS)
        for name in value:
            if name not in known:
                raise ValueError(f"未知的候选模型 {name!r}，可用：{', '.join(sorted(known))}")
        # 去重且保持请求顺序
        return list(dict.fromkeys(value))


class MLPredictRequest(BaseModel):
    """POST /ml/experiments/{id}/predict 的预测请求。

    每一条 record 是 {字段名: 值} 的字典；要求包含训练时的全部特征字段。
    字段顺序由服务端按训练 metadata 重排，多余字段会被忽略并给出 warning。
    """

    records: list[dict[str, Any]] = Field(..., min_length=1, description="待预测记录（最多 1000 条）")


class MLExperimentListItem(BaseModel):
    """GET /ml/experiments 列表项（按 created_at 排序）。"""

    experiment_id: str
    dataset_id: str
    source_version_id: str
    created_at: str | None = None
    target_column: str
    task_type: str
    best_model: str | None = None
    primary_metric: str | None = None
    primary_score: float | None = None
    test_score_summary: dict[str, Any] = Field(default_factory=dict)
    train_rows: int | None = None
    test_rows: int | None = None
    effective_cv_folds: int | None = None
    warnings: list[str] = Field(default_factory=list)
    beats_baseline: bool | None = None
