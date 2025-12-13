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
import re
from pathlib import Path
import threading
import lm_eval
from lm_eval.tasks import TaskManager
# 导入模型相关的函数
from api.models import get_model_args, load_model_from_file

router = APIRouter()

# 全局 TaskManager 实例
task_manager = TaskManager()

# 存储任务状态（使用文件持久化）
TASKS_DIR = Path(__file__).parent.parent.parent / "data" / "tasks"
TASKS_DIR.mkdir(parents=True, exist_ok=True)

tasks_db: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# 结果存储目录
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_task_to_file(task_id: str, task_data: Dict[str, Any]):
    """保存任务数据到文件"""
    task_file = TASKS_DIR / f"{task_id}.json"
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(task_data, f, ensure_ascii=False, indent=2)


def load_task_from_file(task_id: str) -> Optional[Dict[str, Any]]:
    """从文件加载任务数据"""
    task_file = TASKS_DIR / f"{task_id}.json"
    if task_file.exists():
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载任务文件失败 {task_id}: {e}")
            return None
    return None


def load_all_tasks_from_files():
    """从文件加载所有任务"""
    tasks = {}
    for task_file in TASKS_DIR.glob("*.json"):
        task_id = task_file.stem
        try:
            task_data = load_task_from_file(task_id)
            if task_data:
                tasks[task_id] = task_data
        except Exception as e:
            print(f"加载任务文件失败 {task_id}: {e}")
    return tasks


# 启动时加载所有任务
tasks_db = load_all_tasks_from_files()


class DatasetInfo(BaseModel):
    """数据集信息（用于任务创建）"""
    id: Optional[str] = None
    name: str  # 数据集显示名称（文件夹名称）
    task_name: Optional[str] = None  # 正确的任务名称（从 TaskManager 获取，用于评测）
    path: Optional[str] = None
    config_name: Optional[str] = None


class TaskCreateRequest(BaseModel):
    """创建评测任务请求"""
    model_config = {"protected_namespaces": ()}
    
    name: str
    model: str  # 模型类型，如 "openai-chat-completions", "hf" 等
    model_args: Optional[Dict[str, Any]] = None  # 模型参数（如果提供了 model_id，将从本地文件自动构建）
    model_id: Optional[str] = None  # 模型 ID（如果提供，将从本地文件加载模型并自动构建 model_args）
    tasks: List[str]  # 评测任务列表（应该使用正确的任务名称）
    datasets: Optional[List[DatasetInfo]] = None  # 可选的数据集信息，用于验证和提供更多上下文
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
    model: str  # 模型类型
    model_name: Optional[str] = None  # 模型名称（用于显示）
    tasks: List[str]
    datasets: Optional[List[DatasetInfo]] = None # 数据集信息
    created_at: str
    updated_at: str
    progress: Optional[Dict[str, Any]] = None
    result_file: Optional[str] = None
    error_message: Optional[str] = None


def run_evaluation(task_id: str, request: TaskCreateRequest):
    """在后台运行评测任务"""
    import os
    
    try:
        with tasks_lock:
            tasks_db[task_id]["status"] = "running"
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()
            save_task_to_file(task_id, tasks_db[task_id])

        # 确保 model_args 被正确初始化
        if not request.model_args:
            request.model_args = {}
        
        # 如果提供了 model_id，重新构建 model_args（确保 base_url 包含正确路径）
        if request.model_id:
            try:
                model_data = load_model_from_file(request.model_id)
                if model_data:
                    # 使用 get_model_args 重新构建 model_args，确保 base_url 包含正确路径
                    from api.models import get_model_args
                    rebuilt_model_args = get_model_args(model_data)
                    # 合并原有的 model_args（保留用户自定义的参数）
                    rebuilt_model_args.update(request.model_args)
                    request.model_args = rebuilt_model_args
                    print(f"[DEBUG] 从模型文件重新构建 model_args，base_url: {request.model_args.get('base_url')}")
                else:
                    print(f"[WARNING] 无法加载模型文件 {request.model_id}")
            except Exception as e:
                print(f"[ERROR] 加载模型文件 {request.model_id} 失败: {e}")
        
        # 从模型文件或 model_args 中读取 api_key
        original_api_key = None
        api_key = None
        
        # 优先从 model_id 加载 api_key（直接从模型文件读取，确保使用最新的）
        if request.model_id:
            try:
                model_data = load_model_from_file(request.model_id)
                if model_data:
                    if model_data.get("api_key"):
                        api_key = model_data["api_key"]
                        print(f"[DEBUG] 从模型文件 {request.model_id} 读取 api_key: {api_key[:10] if len(api_key) > 10 else '***'}...")
                    else:
                        print(f"[WARNING] 模型文件 {request.model_id} 中没有 api_key 字段")
                else:
                    print(f"[WARNING] 无法加载模型文件 {request.model_id}")
            except Exception as e:
                print(f"[ERROR] 加载模型文件 {request.model_id} 失败: {e}")
        
        # 如果 model_args 中有 api_key，也检查它（但 model_id 的优先级更高）
        if not api_key and "api_key" in request.model_args:
            api_key_from_args = request.model_args["api_key"]
            # 忽略预览文本和无效值
            if (api_key_from_args and 
                api_key_from_args != "(已保存，后端会自动使用)" and 
                api_key_from_args != "(未设置)" and
                isinstance(api_key_from_args, str) and
                len(api_key_from_args) > 10):  # 真实的 api_key 应该比较长
                api_key = api_key_from_args
                print(f"[DEBUG] 从 model_args 读取 api_key: {api_key[:10]}...")
        
        # 确保 model_args 中包含 api_key（用于传递给模型类的 __init__）
        if api_key:
            request.model_args["api_key"] = api_key
            print(f"[DEBUG] 已将 api_key 添加到 model_args")
            # 同时设置环境变量（作为备用，某些模型类可能仍需要）
            original_api_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"[DEBUG] 已设置环境变量 OPENAI_API_KEY")
        else:
            print(f"[WARNING] 未找到 api_key！")
            print(f"[WARNING] model_id={request.model_id}")
            print(f"[WARNING] model_args keys: {list(request.model_args.keys()) if request.model_args else None}")
            if request.model_id:
                try:
                    model_data = load_model_from_file(request.model_id)
                    print(f"[WARNING] 模型文件内容 keys: {list(model_data.keys()) if model_data else None}")
                except Exception as e:
                    print(f"[WARNING] 无法加载模型文件: {e}")
        
        # 准备评测参数
        eval_kwargs = {
            "model": request.model,
            "model_args": request.model_args,
            "tasks": request.tasks,
        }
        
        print(f"[DEBUG] 评测参数 - model: {request.model}, base_url: {request.model_args.get('base_url')}, model_args keys: {list(request.model_args.keys()) if request.model_args else None}")
        
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
        
        # 检查结果是否有效
        if results is None:
            raise ValueError("评测返回 None，可能是多进程环境中的非主进程")
        
        if not isinstance(results, dict):
            raise ValueError(f"评测返回了意外的结果类型: {type(results)}")

        # 保存结果
        result_file = RESULTS_DIR / f"{task_id}_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 更新任务状态
        with tasks_lock:
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result_file"] = str(result_file)
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()
            
            # 安全地提取结果摘要
            results_data = results.get("results", {})
            if not isinstance(results_data, dict):
                results_data = {}
            
            tasks_db[task_id]["progress"] = {
                "completed": True,
                "results_summary": {
                    task: {
                        metric: value
                        for metric, value in task_results.items()
                        if not metric.endswith(",stderr")
                    }
                    for task, task_results in results_data.items()
                    if isinstance(task_results, dict)
                }
            }
            save_task_to_file(task_id, tasks_db[task_id])

    except Exception as e:
        # 恢复原始环境变量（如果之前修改过）
        if original_api_key is not None:
            if original_api_key:
                os.environ["OPENAI_API_KEY"] = original_api_key
            else:
                # 如果原来没有设置，删除环境变量
                os.environ.pop("OPENAI_API_KEY", None)
        
        # 错误信息处理（不再使用名称映射，因为现在应该使用正确的任务名称）
        error_message = str(e)
        
        # 检查是否是 Loglikelihood 不支持 chat completions 的错误
        if "Loglikelihood" in error_message and ("chat completions" in error_message.lower() or "multiple_choice" in error_message.lower()):
            error_message = (
                "错误：当前选择的模型类型（chat completions）不支持 Loglikelihood 类型的任务。\n\n"
                "原因：OpenAI 的 chat completions API 不提供 prompt logprobs，因此无法运行需要 loglikelihood 的任务（如 multiple_choice 类型）。\n\n"
                "解决方案：\n"
                "1. 使用支持 loglikelihood 的模型类型（如 'openai' 而不是 'openai-chat-completions'）\n"
                "2. 或者选择使用 'generate_until' 输出类型的任务（如 mmlu_generative 而不是 mmlu）\n"
                "3. 或者使用其他支持 loglikelihood 的模型（如 HuggingFace 模型）\n\n"
                "相关链接：\n"
                "- https://github.com/EleutherAI/lm-evaluation-harness/issues/942\n"
                "- https://github.com/EleutherAI/lm-evaluation-harness/issues/1196"
            )
        # 如果错误信息只是一个任务名称（如 'mmlu'），添加更多上下文
        # 检查是否是简单的任务名称格式（带引号或不带引号）
        elif re.match(r"^['\"]?([a-zA-Z0-9_/]+)['\"]?$", error_message.strip()):
            task_name = re.match(r"^['\"]?([a-zA-Z0-9_/]+)['\"]?$", error_message.strip()).group(1)
            # 检查这个任务名称是否在请求的任务列表中
            if task_name in request.tasks:
                error_message = f"任务 '{task_name}' 执行失败。请检查任务配置、模型参数或数据集是否正确。"
            else:
                # 可能是任务名称匹配问题
                available_tasks = set(task_manager.all_subtasks)
                available_groups = set(task_manager.all_groups)
                available_all = available_tasks.union(available_groups)
                if hasattr(task_manager, 'all_tasks'):
                    available_all = available_all.union(set(task_manager.all_tasks))
                
                if task_name not in available_all:
                    error_message = f"任务 '{task_name}' 未找到。请确认任务名称是否正确，或使用 '修复任务名称' 功能修复。"
                else:
                    error_message = f"任务 '{task_name}' 执行失败。请检查任务配置、模型参数或数据集是否正确。"
        
        with tasks_lock:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error_message"] = error_message
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()
            save_task_to_file(task_id, tasks_db[task_id])


# 已移除 normalize_task_names 函数，因为现在应该直接使用正确的任务名称（从数据集对象的 task_name 字段获取）


@router.post("/", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest, background_tasks: BackgroundTasks):
    """创建新的评测任务"""
    # 如果提供了 model_id，从本地文件加载模型并构建 model_args
    final_model_args = request.model_args
    model_name = None  # 模型名称（用于显示）
    if request.model_id:
        model_data = load_model_from_file(request.model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail=f"模型不存在: {request.model_id}")
        
        # 获取模型名称
        model_name = model_data.get("name")
        
        # 使用模型数据构建 model_args
        final_model_args = get_model_args(model_data)
        # 如果请求中也提供了 model_args，合并它们（请求中的优先级更高）
        if request.model_args:
            final_model_args.update(request.model_args)
        
        # 确保 model 类型与模型数据一致
        if not request.model or request.model != model_data.get("model_type"):
            # 如果请求中的 model 类型与模型数据不一致，使用模型数据中的类型
            request.model = model_data.get("model_type", request.model)
    
    # 验证 model_args 是否存在
    if not final_model_args:
        raise HTTPException(status_code=400, detail="model_args 是必需的，请提供 model_id 或 model_args")
    
    # 如果提供了数据集信息，优先使用数据集中的 task_name（正确的任务名称）
    if request.datasets and len(request.datasets) > 0:
        # 使用数据集提供的 task_name（从 TaskManager 获取的正确任务名称）
        task_names_from_datasets = [ds.task_name for ds in request.datasets if ds.task_name]
        
        # 优先使用数据集中的 task_name，因为它们应该是正确的
        if task_names_from_datasets:
            normalized_tasks = task_names_from_datasets
        else:
            # 如果数据集中没有 task_name，使用 tasks 列表（前端传递的）
            # 但需要验证这些任务名称是否有效
            normalized_tasks = request.tasks
    else:
        # 如果没有提供数据集信息，直接使用 tasks（前端应该已经传递了正确的名称）
        normalized_tasks = request.tasks
    
    # 检查是否有无效的任务名称
    # 注意：TaskManager 的 all_subtasks 和 all_groups 可能不包含所有有效的任务名称
    # 尤其是对于使用 yaml 配置文件动态加载的任务，或者某些通过 path 加载的任务
    # 因此，如果任务名称在 TaskManager 中找不到，但它看起来像是一个有效的任务（例如 path 格式），我们尝试信任它
    
    available_tasks = set(task_manager.all_subtasks)
    available_groups = set(task_manager.all_groups)
    available_all = available_tasks.union(available_groups)
    
    # 尝试将 task_manager.all_tasks 也加入，以防遗漏
    if hasattr(task_manager, 'all_tasks'):
        available_all = available_all.union(set(task_manager.all_tasks))
    
    invalid_tasks = []
    for t in normalized_tasks:
        if t not in available_all:
            # 尝试更宽松的验证：
            # 1. 如果它在 datasets 列表中有对应的 task_name，我们假设它是有效的（因为 datasets.py 已经验证过）
            # 2. 如果它看起来像是一个路径或者包含配置信息，我们也暂时允许
            
            is_valid_override = False
            
            # 检查 datasets 中的 task_name
            if request.datasets:
                for ds in request.datasets:
                    if ds.task_name == t:
                        is_valid_override = True
                        break
            
            if not is_valid_override:
                # 如果没有被覆盖，才标记为无效
                invalid_tasks.append(t)
    
    if invalid_tasks:
        # 提供更友好的错误信息
        # 但考虑到 TaskManager 可能不完整，这里改为警告并允许继续，或者只记录日志
        # 为了安全起见，我们还是抛出异常，但提供更详细的建议
        error_msg = f"以下任务名称无效或不存在: {', '.join(invalid_tasks)}。\n"
        error_msg += "这可能是因为数据集没有对应的任务名称（task_name）。\n"
        error_msg += "请确保选择的数据集在 TaskManager 中有对应的任务定义。\n"
        error_msg += f"可用的任务名称示例: {', '.join(list(available_tasks)[:10])}..."
        
        # 记录日志而不是直接抛出异常（或者在开发模式下允许）
        # 这里我们选择相信 datasets.py 的判断，如果 task_name 是从 datasets.py 获取的，它应该是正确的
        # 但如果是前端直接传递的字符串，可能是错误的
        
        # 只有当确实找不到任何匹配时才报错
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
    
    task_id = str(uuid.uuid4())
    
    # 使用转换后的任务名称
    task_data = {
        "id": task_id,
        "name": request.name,
        "status": "pending",
        "model": request.model,  # 模型类型
        "model_name": model_name,  # 模型名称（用于显示）
        "model_args": final_model_args,  # 使用构建好的 model_args
        "tasks": normalized_tasks,  # 使用转换后的任务名称
        "datasets": [d.dict() for d in request.datasets] if request.datasets else None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "progress": None,
        "result_file": None,
        "error_message": None,
        "config_id": request.config_id,
        "model_id": request.model_id,  # 保存 model_id 以便后续查询
        "original_request_data": request.dict() # Store the request data
    }

    with tasks_lock:
        tasks_db[task_id] = task_data
        save_task_to_file(task_id, task_data)

    # 创建新的请求对象，使用转换后的任务名称
    request_with_normalized_tasks = TaskCreateRequest(
        name=request.name,
        model=request.model,
        model_args=final_model_args,  # 使用构建好的 model_args
        model_id=request.model_id,
        tasks=normalized_tasks,  # 使用转换后的任务名称
        num_fewshot=request.num_fewshot,
        batch_size=request.batch_size,
        device=request.device,
        limit=request.limit,
        log_samples=request.log_samples,
        apply_chat_template=request.apply_chat_template,
        gen_kwargs=request.gen_kwargs,
        config_id=request.config_id
    )

    # 在后台运行评测（request_with_normalized_tasks 已包含构建好的 model_args）
    background_tasks.add_task(run_evaluation, task_id, request_with_normalized_tasks)

    return TaskResponse(**task_data)


# 已移除 build_task_name_mapping 函数，因为现在应该直接使用正确的任务名称（从数据集对象的 task_name 字段获取）


@router.post("/{task_id}/start")
async def start_task(task_id: str, background_tasks: BackgroundTasks):
    """启动评测任务"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = tasks_db[task_id]
        
        if task["status"] == "running" or task["status"] == "pending":
            raise HTTPException(status_code=400, detail="任务正在运行或等待中，无法重复启动")
        
        # 重置任务状态并重新启动
        task["status"] = "pending"
        task["error_message"] = None
        task["progress"] = None
        task["updated_at"] = datetime.now().isoformat()
        task["result_file"] = None # Clear old results
        save_task_to_file(task_id, task)

        # Re-add to background tasks with original request data
        original_request_data = task["original_request_data"].copy()
        
        # 如果原始请求中有 model_id，重新从模型文件加载 model_args（包括最新的 api_key）
        if original_request_data.get("model_id"):
            model_id = original_request_data["model_id"]
            model_data = load_model_from_file(model_id)
            if model_data:
                # 重新构建 model_args，确保包含最新的 api_key
                final_model_args = get_model_args(model_data)
                # 如果原始请求中也有 model_args，合并它们（原始请求中的优先级更高）
                if original_request_data.get("model_args"):
                    final_model_args.update(original_request_data["model_args"])
                original_request_data["model_args"] = final_model_args
                # 确保 model 类型与模型数据一致
                if not original_request_data.get("model") or original_request_data["model"] != model_data.get("model_type"):
                    original_request_data["model"] = model_data.get("model_type", original_request_data.get("model"))
            else:
                raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
        
        original_request = TaskCreateRequest(**original_request_data)
        background_tasks.add_task(run_evaluation, task_id, original_request)
        
    return {"message": "任务已启动"}


@router.post("/{task_id}/stop")
async def stop_task(task_id: str):
    """终止评测任务"""
    with tasks_lock:
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task = tasks_db[task_id]
        
        if task["status"] != "running":
            raise HTTPException(status_code=400, detail="任务未在运行，无法终止")
        
        task["status"] = "failed" # Mark as failed due to manual stop
        task["error_message"] = "任务被用户手动终止" # Task manually stopped by user
        task["updated_at"] = datetime.now().isoformat()
        save_task_to_file(task_id, task)
        
    return {"message": "任务已终止"}



@router.get("/", response_model=List[TaskResponse])
async def list_tasks():
    """获取所有评测任务列表"""
    # 重新加载文件中的任务（以防有外部修改）
    with tasks_lock:
        file_tasks = load_all_tasks_from_files()
        # 合并文件中的任务到内存中
        for task_id, task_data in file_tasks.items():
            if task_id not in tasks_db or task_data.get("updated_at", "") > tasks_db[task_id].get("updated_at", ""):
                tasks_db[task_id] = task_data
            
            # 如果任务没有 model_name 但有 model_id，尝试从模型文件加载模型名称
            task = tasks_db[task_id]
            
            # 补救措施：如果 datasets 为空，但 original_request_data 中有，尝试恢复
            if not task.get("datasets") and task.get("original_request_data") and task["original_request_data"].get("datasets"):
                task["datasets"] = task["original_request_data"]["datasets"]
                # 可选：保存修复后的数据
                # save_task_to_file(task_id, task)

            if not task.get("model_name") and task.get("model_id"):
                try:
                    model_data = load_model_from_file(task["model_id"])
                    if model_data and model_data.get("name"):
                        task["model_name"] = model_data["name"]
                        # 更新任务数据
                        save_task_to_file(task_id, task)
                except Exception as e:
                    print(f"[WARNING] 无法加载模型名称 {task.get('model_id')}: {e}")
        
        return [TaskResponse(**task) for task in tasks_db.values()]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """获取单个任务详情"""
    with tasks_lock:
        # 如果内存中没有，尝试从文件加载
        if task_id not in tasks_db:
            task_data = load_task_from_file(task_id)
            if task_data:
                tasks_db[task_id] = task_data
            else:
                raise HTTPException(status_code=404, detail="任务不存在")
        
        task = tasks_db[task_id]
        
        # 补救措施：如果 datasets 为空，但 original_request_data 中有，尝试恢复
        if not task.get("datasets") and task.get("original_request_data") and task["original_request_data"].get("datasets"):
            task["datasets"] = task["original_request_data"]["datasets"]
            # 可选：保存修复后的数据
            # save_task_to_file(task_id, task)
        
        # 如果任务没有 model_name 但有 model_id，尝试从模型文件加载模型名称
        if not task.get("model_name") and task.get("model_id"):
            try:
                model_data = load_model_from_file(task["model_id"])
                if model_data and model_data.get("name"):
                    task["model_name"] = model_data["name"]
                    # 更新任务数据
                    save_task_to_file(task_id, task)
            except Exception as e:
                print(f"[WARNING] 无法加载模型名称 {task.get('model_id')}: {e}")
        
        return TaskResponse(**task)


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
        
        # 删除任务文件
        task_file = TASKS_DIR / f"{task_id}.json"
        if task_file.exists():
            os.remove(task_file)
        
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
