"""
大模型评测任务管理平台 - 后端主程序
使用 FastAPI 框架
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api import tasks, datasets, models

app = FastAPI(
    title="大模型评测任务管理平台",
    description="基于 lm-evaluation-harness 的评测任务管理平台 API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(tasks.router, prefix="/api/tasks", tags=["评测任务"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["数据集管理"])
app.include_router(models.router, prefix="/api/models", tags=["模型管理"])


@app.get("/")
async def root():
    """根路径"""
    return {"message": "大模型评测任务管理平台 API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)

