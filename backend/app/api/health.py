"""系统健康检查接口。"""

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health", summary="后端健康检查")
def health_check():
    """用于前后端联通性检查，返回服务运行状态。"""
    return {
        "status": "ok",
        "message": "AI Data Platform backend is running",
    }
