"""AI Data Platform 后端入口。

启动命令（在 backend 目录下）：
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.agent import router as agent_router
from app.api.data import router as data_router
from app.api.datasets import router as datasets_router
from app.api.explainability import router as explainability_router
from app.api.health import router as health_router
from app.api.ml import router as ml_router
from app.api.processing import router as processing_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="智能数据分析与预测平台后端服务（AI Data Intelligence Platform）",
    version="0.7.2",
)

# CORS：允许前端开发服务器跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由：统一挂在 /api 前缀下
app.include_router(health_router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(processing_router, prefix="/api")
app.include_router(ml_router, prefix="/api")
app.include_router(explainability_router, prefix="/api")
app.include_router(agent_router, prefix="/api")


@app.get("/", summary="服务根路径")
def root():
    return {
        "app": settings.app_name,
        "version": "0.7.2",
        "health": "/api/health",
        "upload": "/api/data/upload",
        "eda": "/api/datasets/{dataset_id}/eda?version_id=<optional>",
        "preview": "/api/datasets/{dataset_id}/processing/preview",
        "apply": "/api/datasets/{dataset_id}/processing/apply",
        "versions": "/api/datasets/{dataset_id}/versions",
        "ml_train": "/api/datasets/{dataset_id}/ml/train",
        "ml_experiments": "/api/datasets/{dataset_id}/ml/experiments",
        "ml_explain": "/api/ml/experiments/{experiment_id}/explain",
        "agent_analyze": "/api/agent/analyze",
        "docs": "/docs",
    }
