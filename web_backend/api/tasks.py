"""
评测任务管理 API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
import os
from pathlib import Path
import threading
import lm_eval
from lm_eval.tasks import TaskManager

router = APIRouter()

# 全局 TaskManager 实例
task_manager = TaskManager()

# 存储任务状态（实际应用中应使用数据库）
tasks_db: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# 结果存储目录
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class TaskCreateRequest(BaseModel):
    """创建评测任务请求"""
    model_config = {"protected_namespaces": ()}
    
    name: str
    model: str  # 模型类型，如 "openai-chat-completions", "hf" 等
    model_args: Dict[str, Any]  # 模型参数
    tasks: List[str]  # 评测任务列表
    num_fewshot: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    limit: Optional[int] = None
    log_samples: bool = True
    apply_chat_template: Optional[bool] = False
    gen_kwargs: Optional[Dict[str, Any]] = None
    config_id: Optional[str] = None  # 关联的配置 ID


class TaskResponse(BaseModel):
    """任务响应"""
    id: str
    name: str
    status: str  # pending, running, completed, failed
    model: str
    tasks: List[str]
    created_at: str
    updated_at: str
    progress: Optional[Dict[str, Any]] = None
    result_file: Optional[str] = None
    error_message: Optional[str] = None


def run_evaluation(task_id: str, request: TaskCreateRequest):
    """在后台运行评测任务"""
    try:
        with tasks_lock:
            tasks_db[task_id]["status"] = "running"
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()

        # 准备评测参数
        eval_kwargs = {
            "model": request.model,
            "model_args": request.model_args,
            "tasks": request.tasks,
        }
        
        if request.num_fewshot is not None:
            eval_kwargs["num_fewshot"] = request.num_fewshot
        if request.batch_size is not None:
            eval_kwargs["batch_size"] = request.batch_size
        if request.device is not None:
            eval_kwargs["device"] = request.device
        if request.limit is not None:
            eval_kwargs["limit"] = request.limit
        if request.log_samples is not None:
            eval_kwargs["log_samples"] = request.log_samples
        if request.apply_chat_template is not None:
            eval_kwargs["apply_chat_template"] = request.apply_chat_template
        if request.gen_kwargs is not None:
            eval_kwargs["gen_kwargs"] = request.gen_kwargs

        # 运行评测
        results = lm_eval.simple_evaluate(**eval_kwargs)

        # 保存结果
        result_file = RESULTS_DIR / f"{task_id}_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 更新任务状态
        with tasks_lock:
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result_file"] = str(result_file)
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()
            tasks_db[task_id]["progress"] = {
                "completed": True,
                "results_summary": {
                    task: {
                        metric: value
                        for metric, value in task_results.items()
                        if not metric.endswith(",stderr")
                    }
                    for task, task_results in results.get("results", {}).items()
                }
            }

    except Exception as e:
        with tasks_lock:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error_message"] = str(e)
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()


@router.post("/", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest, background_tasks: BackgroundTasks):
    """创建新的评测任务"""
    task_id = str(uuid.uuid4())
    
    task_data = {
        "id": task_id,
        "name": request.name,
        "status": "pending",
        "model": request.model,
        "model_args": request.model_args,
        "tasks": request.tasks,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "progress": None,
        "result_file": None,
        "error_message": None,
        "config_id": request.config_id,
    }

    with tasks_lock:
        tasks_db[task_id] = task_data

    # 在后台运行评测
    background_tasks.add_task(run_evaluation, task_id, request)

    return TaskResponse(**task_data)


@router.get("/", response_model=List[TaskResponse])
async def list_tasks():
    """获取所有评测任务列表"""
    with tasks_lock:
        return [TaskResponse(**task) for task in tasks_db.values()]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """获取单个任务详情"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        return TaskResponse(**tasks_db[task_id])


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除评测任务"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = tasks_db[task_id]
        
        # 如果任务正在运行，不允许删除
        if task["status"] == "running":
            raise HTTPException(status_code=400, detail="无法删除正在运行的任务")
        
        # 删除结果文件
        if task.get("result_file") and os.path.exists(task["result_file"]):
            os.remove(task["result_file"])
        
        del tasks_db[task_id]
        
    return {"message": "任务已删除"}


@router.get("/{task_id}/results")
async def get_task_results(task_id: str):
    """获取任务评测结果"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = tasks_db[task_id]
        
        if task["status"] != "completed":
            raise HTTPException(status_code=400, detail="任务尚未完成")
        
        result_file = task.get("result_file")
        if not result_file or not os.path.exists(result_file):
            raise HTTPException(status_code=404, detail="结果文件不存在")
        
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        return results


@router.get("/{task_id}/download")
async def download_task_results(task_id: str):
    """下载任务评测结果文件"""
    from fastapi.responses import FileResponse
    
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = tasks_db[task_id]
        
        if task["status"] != "completed":
            raise HTTPException(status_code=400, detail="任务尚未完成")
        
        result_file = task.get("result_file")
        if not result_file or not os.path.exists(result_file):
            raise HTTPException(status_code=404, detail="结果文件不存在")
        
        return FileResponse(
            result_file,
            media_type="application/json",
            filename=f"{task['name']}_results.json"
        )


@router.get("/{task_id}/progress")
async def get_task_progress(task_id: str):
    """获取任务进度"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = tasks_db[task_id]
        
        return {
            "status": task["status"],
            "progress": task.get("progress"),
            "error_message": task.get("error_message"),
            "updated_at": task["updated_at"]
        }


@router.get("/available-tasks/list")
async def get_available_tasks():
    """获取所有可用的评测任务列表"""
    try:
        return {
            "subtasks": task_manager.all_subtasks,
            "groups": task_manager.all_groups,
            "tags": task_manager.all_tags,
            "all_tasks": task_manager.all_tasks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")

