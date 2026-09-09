# AI Data Intelligence Platform

智能数据分析与预测平台（AI Data Intelligence Platform）

一个可长期扩展的 AI 数据智能分析平台。

**当前进度（v0.3.0）**
- ✅ 基础架构：React + Vite 前端 / FastAPI 后端 / Git
- ✅ CSV、Excel(.xlsx/.xls) 数据上传与解析（只读分析，不修改原始数据）
- ✅ 基础数据画像：字段类型推断、缺失统计、唯一值、前 20 行预览、质量警告
- ✅ 临时数据集会话（Dataset Session）：上传即建会话、独立目录、自动过期、可手动删除
- ✅ 自动 EDA：数值统计 / 直方图、分类分布、缺失值分析、IQR 异常值、Pearson 相关、规则型洞察摘要
- ✅ 前端 ECharts 可视化：直方图、分类条形图、缺失热图、异常值箱线、相关性热力图
- ⬜ 后续规划：数据质量检测、特征工程、机器学习建模、模型解释（SHAP）、LLM 分析、Skill / MCP 扩展

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React + Vite + ECharts |
| 后端 | Python 3.14 + FastAPI |
| 数据处理 | pandas、numpy、openpyxl、xlrd |
| 后续扩展 | scikit-learn、XGBoost、SHAP、LLM、Skill、MCP |
| 版本管理 | Git |

## 目录结构

```
AI-Data-Platform/
├── backend/
│   ├── app/
│   │   ├── main.py            # 应用入口（注册 /api 路由）
│   │   ├── api/
│   │   │   ├── health.py      # GET /api/health
│   │   │   ├── data.py        # POST /api/data/upload
│   │   │   └── datasets.py    # GET /api/datasets/{id}/eda、DELETE /api/datasets/{id}
│   │   ├── core/config.py     # 配置（端口、CORS、上传限制等）
│   │   ├── services/
│   │   │   ├── data_service.py     # 文件解析 / 数据画像 / 质量分析
│   │   │   ├── dataset_manager.py  # Dataset Session 的创建 / 读取 / 过期清理 / 删除
│   │   │   └── eda_service.py      # 自动 EDA：统计 / 分布 / 缺失 / 异常 / 相关
│   │   └── models/
│   │       ├── data.py        # 上传响应的 Pydantic 模型
│   │       └── eda.py         # 自动 EDA 响应的 Pydantic 模型
│   ├── runtime/               # 运行时临时数据（会话目录，自动清理，勿提交）
│   ├── tests/                 # pytest 自动化测试
│   └── requirements.txt
├── data/
│   └── sample/demo.csv        # 示例数据（含数值/分类/文本/日期/布尔列、缺失与重复）
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # 首页（系统状态 + 数据上传 + EDA 结果入口）
│   │   ├── upload/UploadPanel.jsx  # 上传与结果展示组件
│   │   ├── eda/               # EDA 可视化面板（ECharts）
│   │   │   ├── EDASection.jsx     # EDA 结果总览容器
│   │   │   ├── HistogramPanel.jsx # 数值分布直方图
│   │   │   ├── CategoricalPanel.jsx
│   │   │   ├── MissingChart.jsx   # 缺失值分析图
│   │   │   ├── OutlierPanel.jsx   # IQR 异常值分析
│   │   │   ├── CorrelationHeatmap.jsx
│   │   │   ├── NumericTable.jsx / EChart.jsx / chartTheme.js
│   │   │   └── ...
│   │   └── ...
│   ├── index.html
│   └── vite.config.js         # /api 代理到后端
├── .env.example
├── .gitignore
└── README.md
```

## 快速开始

### 1. 启动后端（FastAPI，端口 8000）

```powershell
cd backend

# 首次：创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 启动开发服务器
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

- 健康检查：<http://127.0.0.1:8000/api/health>
- 接口文档：<http://127.0.0.1:8000/docs>（Swagger UI）

### 2. 启动前端（React + Vite，端口 5173）

```powershell
cd frontend
npm install      # 首次
npm run dev
```

浏览器访问 <http://localhost:5173>。页面会自动请求 `/api/health` 并显示后端连接状态；下方的“数据上传”区域支持点击或拖拽上传 `.csv / .xlsx / .xls` 文件。上传成功后自动完成两段式分析：

1. **数据画像（v0.2.0）**：数据集概览、字段信息表、质量警告与前 20 行预览；
2. **自动 EDA（v0.3.0）**：基于临时 Dataset Session，展示数值统计与直方图、分类分布、缺失值分析、IQR 异常值分析与 Pearson 相关性热力图等图表。

## API 说明

### GET /api/health

```json
{ "status": "ok", "message": "AI Data Platform backend is running" }
```

### POST /api/data/upload

multipart 表单字段 `file`。仅支持 `.csv / .xlsx / .xls`，最大 20MB（可通过环境变量 `DATA_MAX_UPLOAD_MB` 调整）。

上传成功后会创建一个临时 **Dataset Session**（源文件与画像结果写入 `backend/runtime/datasets/`，默认有效期 `DATASET_TTL_HOURS` 小时），后续 EDA 接口基于该会话 ID 读取数据。

成功返回结构：

```json
{
  "success": true,
  "dataset_id": "9f6c…-uuid",
  "created_at": "2026-09-09T12:00:00Z",
  "expires_at": "2026-09-09T20:00:00Z",
  "dataset": { "file_name": "demo.csv", "file_type": "csv", "file_size": 2345, "rows": 23, "columns": 7 },
  "quality": { "total_missing": 7, "columns_with_missing": 3, "duplicate_rows": 1, "empty_columns": 0, "constant_columns": 0 },
  "column_profiles": [
    { "column_name": "age", "dtype": "float64", "inferred_type": "numeric", "non_null_count": 22, "missing_count": 1, "missing_percentage": 4.35, "unique_count": 20 }
  ],
  "preview": [ { "id": 1, "city": "北京", "score": 88.5, "joined": "2024-01-15", "is_active": "true", "notes": "..." } ],
  "warnings": ["发现 3 个包含缺失值的字段（共 7 个缺失单元格）", "发现 1 行重复数据"]
}
```

`inferred_type` 依据数据内容推断，取值包括：`numeric`、`categorical`、`datetime`、`boolean`、`text`。

失败返回结构（HTTP 400/413/422/500）：

```json
{ "success": false, "error": { "code": "empty_file", "message": "上传的文件内容为空，请检查文件后重试。" } }
```

常见错误码：`empty_file`、`unsupported_file_type`、`file_too_large`、`csv_encoding_error`、`csv_parse_error`、`excel_parse_error`、`no_rows`、`internal_error`。

### GET /api/datasets/{dataset_id}/eda

对已上传的数据集会话执行自动 EDA，返回结构化 JSON（响应结构由 `models/eda.py` 显式定义）：

```json
{
  "success": true,
  "dataset_id": "…",
  "dataset": { "file_name": "demo.csv", "rows": 23, "columns": 7 },
  "summary": {
    "total_rows": 23, "total_columns": 7,
    "numeric_columns": 2, "categorical_columns": 3,
    "datetime_columns": 1, "text_columns": 1,
    "columns_with_outliers": 1, "strong_correlations": 1,
    "rule_based_insights": ["…强相关洞察 / 缺失提示…"]
  },
  "numeric_summaries": [{ "column": "age", "count": 22, "missing_count": 1, "mean": 33.2, "median": 31.0, "iqr": 15.0, "skewness": 0.8, "outlier_count": 1 }],
  "numeric_distributions": [{ "column": "age", "bin_edges": [0, 10, 20], "counts": [2, 5, 8] }],
  "categorical_summaries": [{ "column": "city", "unique_count": 5, "top_values": [{ "value": "北京", "count": 8, "percentage": 34.8 }] }],
  "missing_analysis": { "total_missing": 7, "total_cells": 161, "overall_missing_percentage": 4.3, "by_column": [] },
  "outlier_analysis": [{ "column": "age", "detection_method": "IQR", "outlier_count": 2, "lower_bound": -2.5, "upper_bound": 57.5 }],
  "correlation": { "columns": ["age", "score"], "matrix": [[1.0, 0.83], [0.83, 1.0]], "top_correlations": [] }
}
```

主要能力：
- **数值列**：均值/标准差/四分位数/极差/IQR/偏度/零值数 + 直方图分箱数据；
- **分类/布尔/低基数文本列**：频次 TopN（含占比）；
- **缺失值**：整体缺失率与逐列明细（缺失率降序）；
- **异常值**：IQR 法（Q1 − 1.5·IQR / Q3 + 1.5·IQR）；
- **相关性**：数值列 Pearson 相关矩阵 + `|r|` 降序 Top 对（规模上限保护）；
- **摘要**：字段构成统计与规则型洞察（纯规则计算，未接入 LLM）。

> 设计约定：EDA 只做**只读分析**；分布图以 JSON 分箱数据下发，图表在前端用 ECharts 渲染，后端不生成图片文件。

### DELETE /api/datasets/{dataset_id}

删除临时 Dataset Session（源文件与 metadata 一并清理）：

```json
{ "success": true, "message": "Dataset session deleted" }
```

会话类错误码：`dataset_not_found`、`dataset_expired`、`invalid_dataset_id`、`dataset_delete_failed`。

## 自动化测试

```powershell
cd backend
.venv\Scripts\python.exe -m pytest tests -v
```

覆盖：CSV 成功 / Excel(.xlsx) 成功 / 非法扩展名 / 空文件 / 损坏 CSV / 损坏 Excel / 超限文件 / 健康检查；Dataset Session 上传建会话、落盘、删除 / EDA 结构与数值统计 / 缺失 / IQR 异常 / Pearson 相关 / 分类 TopN / 非法、不存在与过期 dataset_id / 边界数据。

## 手动测试上传

```powershell
# PowerShell（简单）
Invoke-RestMethod http://127.0.0.1:8000/api/health

# curl 上传示例文件（响应中的 dataset_id 供 EDA 使用）
curl.exe -F "file=@data/sample/demo.csv" http://127.0.0.1:8000/api/data/upload

# curl 对上传的会话执行自动 EDA（替换为上面返回的 dataset_id）
curl.exe http://127.0.0.1:8000/api/datasets/9f6c…-uuid/eda

# 或直接用浏览器打开 http://localhost:5173 拖拽上传，自动触发画像 + EDA 图表
```

## 数据处理流程

1. 路由层（`api/data.py`）：读取 multipart 文件（分块读取并限制 20MB）→ 只保留文件名
2. 校验与解析层（`services/data_service.py`）：空文件 / 扩展名 / 大小 → pandas 解析（CSV 自动尝试 UTF-8/GB18030 编码，Excel 按格式选择 openpyxl/xlrd）
3. 画像层：逐列画像（类型推断、缺失、唯一值）→ 汇总质量概览 → 生成用户可读 warnings → 生成 JSON 安全的前 20 行预览
4. 会话层（`services/dataset_manager.py`）：为本次上传创建临时 Dataset Session（UUID），将源文件与 metadata 落盘到 `backend/runtime/datasets/<dataset_id>/`，返回 `dataset_id`、`created_at`、`expires_at`
5. EDA 层（`services/eda_service.py`）：`GET /api/datasets/{dataset_id}/eda` 时按 ID 加载会话数据，计算数值统计与直方图分箱、分类 TopN、缺失率、IQR 异常值、Pearson 相关矩阵与规则型洞察；全部结果以 JSON 返回
6. 响应模型（`models/data.py` / `models/eda.py`）保证 JSON 结构稳定；任何异常都以结构化 `{ success, error }` 返回，不泄露 traceback
7. 清理：会话默认 2 小时过期，在访问/上传时惰性清理过期目录，也可 `DELETE /api/datasets/{dataset_id}` 手动删除

**设计约定：所有流程均为只读分析，不修改、清洗或覆盖用户原始数据；图表由前端 ECharts 渲染，后端不生成图片。**

## 环境变量

复制 `.env.example` 为 `.env`（项目根目录）后按需修改：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BACKEND_HOST` | `0.0.0.0` | 后端监听地址 |
| `BACKEND_PORT` | `8000` | 后端端口 |
| `BACKEND_CORS_ORIGINS` | `http://localhost:5173` | 允许跨域来源 |
| `DATA_MAX_UPLOAD_MB` | `20` | 上传大小上限（MB） |
| `DATASET_SESSION_TTL_MINUTES` | `120` | Dataset Session 有效期（分钟） |
| `DATASET_RUNTIME_DIR` | `backend/runtime/datasets` | 会话数据存放目录 |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite `/api` 代理目标 |

> 提示：代理目标建议使用 `127.0.0.1`，避免 `localhost` 被解析为 IPv6(`::1`) 而后端仅监听 IPv4 导致 502。

## 路线图

- [x] 项目基础架构与健康检查（v0.1.0）
- [x] 数据上传、解析、基础画像、质量概览、数据预览（v0.2.0）
- [x] Dataset Session 与会话管理（v0.3.0）
- [x] EDA 探索性数据分析（数值/分类/缺失/异常/相关性，v0.3.0）
- [ ] 数据质量检测
- [ ] 特征工程
- [ ] 机器学习建模
- [ ] 模型解释（SHAP）
- [ ] LLM 智能分析
- [ ] Skill / MCP 扩展
