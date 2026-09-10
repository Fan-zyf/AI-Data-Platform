"""v0.5 ML 预处理流水线工厂（防止数据泄露的核心）。

正式机器学习评估中的所有统计型 preprocessing 都必须放在 sklearn Pipeline +
ColumnTransformer 中，并只在训练集 / 每个 CV fold 的训练部分内 fit：

    Numeric Pipeline:      SimpleImputer(median) -> StandardScaler
    Categorical Pipeline:  SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")

- OneHotEncoder 只在训练数据上 fit，测试集 / 预测时出现未见类别通过
  handle_unknown="ignore" 静默映射到全零向量，绝不报错；
- 全流程不拼接 / 不 eval 任何代码，全部由 sklearn 组件执行；
- 预测阶段复用已保存的完整 Pipeline（raw features -> preprocessing -> model），不再重新 fit。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ColumnTransformer 中预处理模块的固定名称（feature_names_out 提取依赖它）
PREPROCESSING_STEP = "preprocessing"


class DtypeNormalizer(BaseEstimator, TransformerMixin):
    """Pipeline 最前端的无状态类型规整器。

    训练与预测都经过它，保证送入下游的列类型完全一致：

    - numeric 列：pd.to_numeric + float64（字符串数字、JSON 中的数值都能统一）；
    - categorical 列：布尔值 / 数值标签 / 文本统一转成字符串（缺失保留为 NaN），
      避免 bool/int 混合类别在 SimpleImputer / OneHotEncoder 下的类型歧义。

    fit 不学习任何统计参数，因此不会造成数据泄露。
    """

    def __init__(self, numeric_cols: list[str] | None = None, categorical_cols: list[str] | None = None):
        self.numeric_cols = list(numeric_cols or [])
        self.categorical_cols = list(categorical_cols or [])

    def fit(self, X, y=None):  # noqa: N803 - sklearn 约定参数名
        return self

    def transform(self, X):  # noqa: N803 - sklearn 约定参数名
        out = X.copy()
        for col in self.numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        for col in self.categorical_cols:
            out[col] = out[col].map(lambda value: np.nan if pd.isna(value) else str(value))
        return out


def _make_one_hot_encoder() -> OneHotEncoder:
    """兼容不同 sklearn 版本创建 OneHotEncoder（dense 输出）。"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - 旧版本 sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore[call-arg]


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """构造「类型规整 + ColumnTransformer」预处理 Pipeline（不含模型）。"""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, list(numeric_cols)))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, list(categorical_cols)))

    steps = [("dtype", DtypeNormalizer(numeric_cols, categorical_cols))]
    steps.append(
        (
            PREPROCESSING_STEP,
            ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                verbose_feature_names_out=True,
            ),
        )
    )
    return Pipeline(steps)


def extract_feature_names(pipeline: Pipeline) -> list[str]:
    """尝试从已 fit 的完整 Pipeline 中提取输出特征名（如 numeric__age）。

    v0.6 SHAP 将复用这些名字。提取失败时返回空列表（由调用方降级为 warning），
    绝不能导致训练失败。
    """
    try:
        step = pipeline.named_steps[PREPROCESSING_STEP]
        # 兼容两种组织方式：外层直接是 ColumnTransformer，
        # 或外层包着「类型规整 + ColumnTransformer」的内层 Pipeline
        if hasattr(step, "named_steps") and PREPROCESSING_STEP in step.named_steps:
            step = step.named_steps[PREPROCESSING_STEP]
        names = step.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:  # noqa: BLE001 - 提取失败不致命
        return []
