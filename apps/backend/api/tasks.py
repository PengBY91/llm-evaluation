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
import multiprocessing
import lm_eval
# from lm_eval.tasks import TaskManager  # Moved to lazy loading
# 导入模型相关的函数
from api.models import get_model_args, load_model_from_file

router = APIRouter()

# 全局 TaskManager 实例（懒加载）
_task_manager = None
_task_manager_lock = threading.Lock()

# 缓存文件路径
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
TASK_MANAGER_CACHE_FILE = CACHE_DIR / "task_manager_cache.json"


class CachedTaskManager:
    """轻量级 TaskManager，从 JSON 缓存加载，避免扫描 YAML 文件"""
    
    def __init__(self, cache_data: dict):
        self.all_subtasks = cache_data.get("all_subtasks", [])
        self.all_groups = cache_data.get("all_groups", [])
        self.all_tags = cache_data.get("all_tags", [])
        self.all_tasks = cache_data.get("all_tasks", [])
        self._cached_at = cache_data.get("cached_at", "unknown")
    
    def __repr__(self):
        return f"CachedTaskManager(subtasks={len(self.all_subtasks)}, groups={len(self.all_groups)}, cached_at={self._cached_at})"


def get_task_manager():
    """同步获取 TaskManager（优先从缓存加载）"""
    global _task_manager
    with _task_manager_lock:
        if _task_manager is None:
            # 优先尝试从缓存加载
            if TASK_MANAGER_CACHE_FILE.exists():
                try:
                    import json
                    print("[DEBUG] Loading TaskManager from cache...")
                    with open(TASK_MANAGER_CACHE_FILE, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    _task_manager = CachedTaskManager(cache_data)
                    print(f"[DEBUG] TaskManager loaded from cache: {_task_manager}")
                except Exception as e:
                    print(f"[WARNING] Failed to load TaskManager cache: {e}")
                    _task_manager = None
            
            # 如果缓存不存在或加载失败，使用完整初始化
            if _task_manager is None:
                print("[DEBUG] Initializing TaskManager (this might take a while)...")
                print("[TIP] 运行 'python scripts/cache_task_manager.py' 可以预缓存，加速后续启动")
                from lm_eval.tasks import TaskManager
                _task_manager = TaskManager()
                print("[DEBUG] TaskManager initialized.")
    return _task_manager


async def get_task_manager_async():
    """异步获取 TaskManager（用于 async 路由，不阻塞事件循环）"""
    import asyncio
    
    global _task_manager
    if _task_manager is not None:
        return _task_manager
    
    # 在线程池中运行初始化，避免阻塞事件循环
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_task_manager)

# 存储任务状态（使用文件持久化）
TASKS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "tasks"
TASKS_DIR.mkdir(parents=True, exist_ok=True)

tasks_db: Dict[str, Dict[str, Any]] = {}
tasks_lock = threading.Lock()

# 结果存储目录
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 需要 loglikelihood 的输出类型
LOGLIKELIHOOD_OUTPUT_TYPES = {"loglikelihood", "loglikelihood_rolling", "multiple_choice"}

# 常用任务的生成式变体映射（用于不支持 loglikelihood 的模型）
# 这些任务使用 generate_until 输出类型，通过正则匹配提取答案，无需 logprobs
GENERATIVE_TASK_ALTERNATIVES = {
    # MMLU 系列
    "mmlu": "mmlu_generative",
    # GPQA 系列
    "gpqa": "gpqa_diamond_generative_n_shot",
    "gpqa_diamond": "gpqa_diamond_generative_n_shot",
    "gpqa_main": "gpqa_main_generative_n_shot",
    "gpqa_extended": "gpqa_extended_generative_n_shot",
    # TruthfulQA
    "truthfulqa_mc1": "truthfulqa_gen",
    "truthfulqa_mc2": "truthfulqa_gen",
    # ARC (需要 arc_challenge_chat.yaml)
    "arc_challenge": "arc_challenge_chat",
}


def get_generative_alternative(task_name: str) -> Optional[str]:
    """
    获取任务的生成式变体（如果存在）
    
    :param task_name: 原始任务名称
    :return: 生成式变体名称，如果不存在则返回 None
    """
    return GENERATIVE_TASK_ALTERNATIVES.get(task_name.lower())


def get_task_output_types(task_names: List[str]) -> Dict[str, str]:
    """
    获取任务列表中每个任务的 output_type
    
    :param task_names: 任务名称列表
    :return: 任务名称到输出类型的映射
    """
    from lm_eval import utils
    
    output_types = {}
    tm = get_task_manager()
    
    for task_name in task_names:
        try:
            # 尝试从 TaskManager 获取任务配置
            if hasattr(tm, '_get_config'):
                config = tm._get_config(task_name)
                if config and 'output_type' in config:
                    output_types[task_name] = config['output_type']
                    continue
            
            # 如果是 CachedTaskManager，尝试从 YAML 文件直接读取
            if hasattr(tm, 'task_index') and task_name in tm.task_index:
                yaml_path = tm.task_index[task_name]
                if yaml_path and yaml_path != -1:
                    try:
                        config = utils.load_yaml_config(yaml_path, mode="full")
                        if config and 'output_type' in config:
                            output_types[task_name] = config['output_type']
                            continue
                    except Exception:
                        pass
            
            # 默认: 如果任务名包含 'generative' 或 'gen'，假设是 generate_until
            if 'generative' in task_name.lower() or '_gen' in task_name.lower():
                output_types[task_name] = 'generate_until'
            else:
                # 常见的 loglikelihood 任务
                loglikelihood_task_patterns = [
                    'mmlu', 'hellaswag', 'arc', 'winogrande', 'piqa', 
                    'lambada', 'sciq', 'boolq', 'openbookqa', 'copa',
                    'rte', 'wsc', 'wic', 'multirc', 'record', 'cb',
                    'storycloze', 'swag', 'siqa', 'truthfulqa_mc'
                ]
                is_loglikelihood = any(
                    pattern in task_name.lower() 
                    for pattern in loglikelihood_task_patterns
                )
                if is_loglikelihood:
                    output_types[task_name] = 'multiple_choice'
                else:
                    output_types[task_name] = 'unknown'
                    
        except Exception as e:
            print(f"[DEBUG] 无法获取任务 '{task_name}' 的 output_type: {e}")
            output_types[task_name] = 'unknown'
    
    return output_types


def requires_loglikelihood(task_names: List[str]) -> bool:
    """
    判断任务列表中是否有任务需要 loglikelihood
    
    :param task_names: 任务名称列表
    :return: 是否需要 loglikelihood
    """
    output_types = get_task_output_types(task_names)
    
    for task_name, output_type in output_types.items():
        if output_type in LOGLIKELIHOOD_OUTPUT_TYPES:
            print(f"[DEBUG] 任务 '{task_name}' 需要 loglikelihood (output_type={output_type})")
            return True
    
    return False


def switch_to_completions_api(model_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 chat completions API 切换为 completions API
    
    :param model_args: 原始模型参数
    :return: 修改后的模型参数
    """
    new_args = model_args.copy()
    
    if 'base_url' in new_args:
        base_url = new_args['base_url']
        # 将 /chat/completions 替换为 /completions
        if '/chat/completions' in base_url:
            new_args['base_url'] = base_url.replace('/chat/completions', '/completions')
            print(f"[INFO] 自动切换 API 端点: {base_url} -> {new_args['base_url']}")
        elif '/v1' in base_url and not base_url.endswith('/completions'):
            # 如果是 /v1 结尾，添加 /completions
            if base_url.endswith('/v1'):
                new_args['base_url'] = base_url + '/completions'
            else:
                new_args['base_url'] = base_url.rstrip('/') + '/completions'
            print(f"[INFO] 自动添加 completions 端点: {base_url} -> {new_args['base_url']}")
    
    return new_args


def patch_tokenizer():
    """解决部分 lm-eval 版本中对 transformers 分词器调用不存在的 encode_batch 方法的 Bug"""
    try:
        from transformers import PreTrainedTokenizerFast
        if not hasattr(PreTrainedTokenizerFast, 'encode_batch'):
            def encode_batch(self, texts, **kwargs):
                # 转换输入确保是列表
                if not isinstance(texts, (list, tuple)):
                    try:
                        texts = list(texts)
                    except:
                        pass
                
                # 核心修复：确保所有元素都是字符串 (处理 JsonChatStr 等对象)
                texts = [str(t) for t in texts]
                
                # 使用单独的 encode 调用，直接返回 token IDs 列表
                result = []
                for text in texts:
                    ids = self.encode(text, **kwargs)
                    result.append(ids)
                return result
            
            PreTrainedTokenizerFast.encode_batch = encode_batch
            print("[DEBUG] 已成功为 PreTrainedTokenizerFast 补全 encode_batch 方法 (返回原始 token IDs)")
    except Exception as e:
        print(f"[DEBUG] 分词器补丁跳过或失败: {e}")

# 执行补丁
patch_tokenizer()


def patch_aiohttp_proxy():
    """强制 lm-eval 的 aiohttp 客户端使用系统代理"""
    try:
        # 检查是否有代理配置
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        proxy_url = https_proxy or http_proxy
        
        if not proxy_url:
            return
        
        print(f"[DEBUG] 正在为 aiohttp 配置代理: {proxy_url}")
        
        # 导入 aiohttp 并 patch ClientSession
        import aiohttp
        original_init = aiohttp.ClientSession.__init__
        
        def patched_init(self, *args, **kwargs):
            # 如果没有显式设置 connector，添加代理支持
            if 'connector' not in kwargs:
                # 创建支持代理的 connector
                connector = aiohttp.TCPConnector(
                    ssl=False,  # 禁用 SSL 验证以避免代理问题
                    force_close=True
                )
                kwargs['connector'] = connector
            
            # 设置信任环境变量中的代理
            if 'trust_env' not in kwargs:
                kwargs['trust_env'] = True
            
            # 增加超时时间
            if 'timeout' not in kwargs:
                timeout = aiohttp.ClientTimeout(total=180, connect=60, sock_read=60)
                kwargs['timeout'] = timeout
            
            return original_init(self, *args, **kwargs)
        
        aiohttp.ClientSession.__init__ = patched_init
        print("[DEBUG] 已成功为 aiohttp.ClientSession 配置代理支持 (trust_env=True)")
        
    except Exception as e:
        print(f"[DEBUG] aiohttp 代理补丁失败: {e}")

# 执行代理补丁
patch_aiohttp_proxy()


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
print("[DEBUG] Loading tasks from files...")
tasks_db = load_all_tasks_from_files()
print(f"[DEBUG] Loaded {len(tasks_db)} tasks.")


class DatasetInfo(BaseModel):
    """数据集信息（用于任务创建）"""
    id: Optional[str] = None
    name: str  # 数据集显示名称（文件夹名称）
    task_name: Optional[str] = None  # 正确的任务名称（从 TaskManager 获取，用于评测）
    path: Optional[str] = None
    config_name: Optional[str] = None
    subtasks: Optional[List[str]] = None  # 子任务列表（如果是 group）


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
    results: Optional[Dict[str, Any]] = None  # 完整的评测结果（仅在 completed 状态且请求详情时返回）


def run_evaluation_process_wrapper(task_id: str, request: TaskCreateRequest):
    """
    Process wrapper for run_evaluation to ensure process isolation.
    This prevents os.environ changes from affecting other tasks and avoids GIL blocking.
    """
    p = multiprocessing.Process(target=run_evaluation, args=(task_id, request))
    p.start()
    # We don't join() here because we want it to run in background.
    # The process will exit when run_evaluation finishes.


def run_evaluation(task_id: str, request: TaskCreateRequest):
    """在后台运行评测任务"""
    import os
    import logging
    from pathlib import Path
    
    # 设置缓存目录到项目根目录（便于离线部署）
    project_root = Path(__file__).parent.parent.parent.parent
    cache_dir = project_root / "outputs" / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # 创建日志目录和任务日志文件
    logs_dir = project_root / "outputs" / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"task_{task_id}.log"
    
    # 配置任务专用 logger
    task_logger = logging.getLogger(f"evaluation.{task_id}")
    task_logger.setLevel(logging.DEBUG)
    
    # 清除已有的 handlers（防止重复添加）
    task_logger.handlers.clear()
    
    # 文件 handler - 保存详细日志到文件
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    task_logger.addHandler(file_handler)
    
    # 控制台 handler - 打印到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)
    task_logger.addHandler(console_handler)
    
    task_logger.info(f"=" * 60)
    task_logger.info(f"开始评测任务: {request.name}")
    task_logger.info(f"任务 ID: {task_id}")
    task_logger.info(f"日志文件: {log_file}")
    task_logger.info(f"=" * 60)
    
    # 捕获标准输出和错误输出到日志
    import sys
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = ""
        
        def write(self, message):
            if message:
                self.buffer += message
                if "\n" in self.buffer:
                    lines = self.buffer.split("\n")
                    for line in lines[:-1]:
                        if line.strip():
                            self.logger.log(self.level, line.strip())
                    self.buffer = lines[-1]
        
        def flush(self):
            if self.buffer.strip():
                self.logger.log(self.level, self.buffer.strip())
                self.buffer = ""

    # 保存原始 stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 重定向到 logger
    sys.stdout = LoggerWriter(task_logger, logging.INFO)
    sys.stderr = LoggerWriter(task_logger, logging.ERROR)
    
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "huggingface" / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "huggingface" / "transformers")
    
    # 设置离线模式环境变量，确保不访问网络下载数据集和指标 (除非用户显式允许)
    if "HF_DATASETS_OFFLINE" not in os.environ:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if "TRANSFORMERS_OFFLINE" not in os.environ:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    task_logger.info(f"环境配置完成 - 离线模式: HF_DATASETS_OFFLINE={os.environ.get('HF_DATASETS_OFFLINE')}")
    task_logger.info(f"缓存目录: {cache_dir / 'huggingface'}")
    
    try:
        with tasks_lock:
            tasks_db[task_id]["status"] = "running"
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()
            tasks_db[task_id]["log_file"] = str(log_file)  # 保存日志文件路径
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
        # ========== 自动处理 DeepSeek 等 Chat 模型对 Loglikelihood 任务的支持问题 ==========
        # DeepSeek 和其他部分 Chat 模型不支持 legacy completions 接口，也无法很好的支持 loglikelihood
        # 因此，我们需要将这些任务替换为生成式版本（如 mmlu -> mmlu_flan_generative）
        
        is_deepseek = False
        
        # Check base_url
        if request.model_args and "base_url" in request.model_args:
            base_url = str(request.model_args["base_url"]).lower()
            if "deepseek" in base_url:
                is_deepseek = True
                
        # Check model name in args
        if not is_deepseek and request.model_args and "model" in request.model_args:
            model_name_arg = str(request.model_args["model"]).lower()
            if "deepseek" in model_name_arg:
                is_deepseek = True

        # Check model configuration
        if request.model_id:
            try:
                model_data = load_model_from_file(request.model_id)
                if model_data:
                    # Check API model name
                    if "deepseek" in str(model_data.get("model_name", "")).lower():
                        is_deepseek = True
                    # Check user-defined display name
                    if "deepseek" in str(model_data.get("name", "")).lower():
                        is_deepseek = True
            except Exception:
                pass

        if is_deepseek:
            print("[INFO] 检测到 DeepSeek 模型，正在检查是否有需要替换的任务...")
            tasks_modified = False
            new_tasks = []
            for task in request.tasks:
                alt_task = get_generative_alternative(task)
                if alt_task:
                    print(f"[INFO] 将任务 '{task}' 替换为生成式变体 '{alt_task}' (适配 DeepSeek)")
                    task_logger.info(f"自动替换任务: {task} -> {alt_task} (适配 DeepSeek Chat API)")
                    new_tasks.append(alt_task)
                    tasks_modified = True
                else:
                    new_tasks.append(task)
            
            if tasks_modified:
                request.tasks = new_tasks
                task_logger.info(f"替换后的任务列表: {request.tasks}")
        
        # ========== 自动切换模型类型（关键逻辑）==========
        # 检测任务是否需要 loglikelihood，如果需要，自动切换到支持的模型类型
        original_model_type = request.model
        needs_loglikelihood = requires_loglikelihood(request.tasks)
        
        # 获取后端类型（从 model_id 或 model_args）
        backend_type = None
        if request.model_id:
            try:
                model_data = load_model_from_file(request.model_id)
                if model_data:
                    backend_type = model_data.get("backend_type")
            except Exception:
                pass
        
        # 判断是否为本地 API 服务（非官方 OpenAI）
        base_url = request.model_args.get("base_url", "")
        is_local_api = backend_type == "openai-api" or (
            base_url and 
            "api.openai.com" not in base_url and
            "azure.com" not in base_url
        )
        
        if needs_loglikelihood:
            if request.model in ["openai-chat-completions", "local-chat-completions"]:
                if is_local_api:
                    # 本地 OpenAI 兼容服务（vLLM, Ollama 等）完全支持 loglikelihood
                    task_logger.info("检测到任务需要 loglikelihood，自动切换模型类型: -> local-completions")
                    print("[INFO] 检测到任务需要 loglikelihood，使用 local-completions（本地 API 支持 echo + logprobs）")
                    
                    request.model = "local-completions"
                    request.model_args = switch_to_completions_api(request.model_args)
                    
                    task_logger.info(f"模型类型已切换: {original_model_type} -> {request.model}")
                    task_logger.info(f"API 端点: {request.model_args.get('base_url')}")
                else:
                    # 官方 OpenAI API：只有 babbage-002 和 davinci-002 支持
                    task_logger.warning("检测到任务需要 loglikelihood，但使用的是官方 OpenAI API")
                    task_logger.warning("官方 OpenAI 仅 babbage-002/davinci-002 支持 loglikelihood")
                    print("[WARNING] 官方 OpenAI Chat API 不支持 loglikelihood，尝试切换到 openai-completions...")
                    
                    request.model = "openai-completions"
                    request.model_args = switch_to_completions_api(request.model_args)
                    
                    task_logger.info(f"模型类型已切换: {original_model_type} -> {request.model}")
                    task_logger.info(f"API 端点: {request.model_args.get('base_url')}")
        elif request.model == "openai-chat-completions" and is_local_api:
            # 对于不需要 loglikelihood 的任务，本地 API 使用 local-chat-completions
            task_logger.info("本地 API 服务，使用 local-chat-completions 模型类型")
            request.model = "local-chat-completions"
        
        # DeepSeek 强制修复：如果是 DeepSeek 且启用了 chat template，必须使用 chat 模型类型
        if is_deepseek and "completions" in request.model and "chat" not in request.model:
            task_logger.info("检测到 DeepSeek 模型使用了错误的 openai-completions 类型，强制切换为 local-chat-completions")
            request.model = "local-chat-completions"
            
            # 修复 URL (如果有必要)
            if "base_url" in request.model_args:
                base_url = request.model_args["base_url"]
                if base_url.endswith("/completions/completions"):
                     request.model_args["base_url"] = base_url.replace("/completions/completions", "/chat/completions")
                elif base_url.endswith("/completions") and "chat" not in base_url:
                     request.model_args["base_url"] = base_url.replace("/completions", "/chat/completions")
                print(f"[INFO] DeepSeek 强制切换: model={request.model}, base_url={request.model_args['base_url']}")
        
        # 准备评测参数
        eval_kwargs = {
            "model": request.model,
            "model_args": request.model_args,
            "tasks": request.tasks,
        }


        # 针对 API 模型自动优化并发参数
        if request.model in ["openai-chat-completions", "openai-completions", "local-completions", "local-chat-completions"]:
            # 如果没有显式设置 num_concurrent，则设置默认值以提高并发性能
            # 默认值设为 100，这对于大多数现代 API 提供商（OpenAI, DeepSeek, vLLM 等）都是理想的并发值
            if "num_concurrent" not in request.model_args:
                default_concurrent = 1 # 降低默认并发数，避免 429/500 Errors
                request.model_args["num_concurrent"] = default_concurrent
                print(f"[INFO] 自动优化: 为 API 模型设置 num_concurrent={default_concurrent}")
            
            # 配置代理支持：确保 aiohttp 能够使用系统代理
            # 检查环境变量中是否有代理配置
            http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            
            if https_proxy or http_proxy:
                proxy_url = https_proxy or http_proxy
                print(f"[INFO] 检测到代理配置: {proxy_url}")
                # 将代理信息传递给 model_args，lm-eval 的某些版本支持这个参数
                # 注意：这取决于 lm-eval 的版本，如果不支持，我们需要通过环境变量传递
                if "aiohttp_client_timeout" not in request.model_args:
                    # 增加超时时间以适应代理环境
                    request.model_args["aiohttp_client_timeout"] = 120
                    print(f"[INFO] 设置 aiohttp 超时为 120 秒以适应代理环境")
        
        task_logger.info(f"模型类型: {request.model}")
        task_logger.info(f"模型 API 地址: {request.model_args.get('base_url')}")
        task_logger.debug(f"model_args 参数: {list(request.model_args.keys()) if request.model_args else None}")
        
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
        # DeepSeek 特殊处理：Chat 模型必须启用 chat template
        if is_deepseek:
            print("[INFO] 检测到 DeepSeek 模型，自动启用 apply_chat_template=True")
            eval_kwargs["apply_chat_template"] = True
        elif request.apply_chat_template is not None:
            eval_kwargs["apply_chat_template"] = request.apply_chat_template
        if request.gen_kwargs is not None:
            eval_kwargs["gen_kwargs"] = request.gen_kwargs

        # 运行评测
        task_logger.info(f"开始运行评测，任务列表: {request.tasks}")
        task_logger.info(f"其他参数: num_fewshot={request.num_fewshot}, batch_size={request.batch_size}, limit={request.limit}")
        
        import time as _time
        start_time = _time.time()
        results = lm_eval.simple_evaluate(**eval_kwargs)
        elapsed_time = _time.time() - start_time
        task_logger.info(f"评测完成，耗时: {elapsed_time:.2f} 秒")
        
        # 检查结果是否有效
        if results is None:
            raise ValueError("评测返回 None，可能是多进程环境中的非主进程")
        
        if not isinstance(results, dict):
            raise ValueError(f"评测返回了意外的结果类型: {type(results)}")

        # 保存结果
        result_file = RESULTS_DIR / f"{task_id}_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        task_logger.info(f"结果已保存到: {result_file}")
        
        # 记录结果摘要
        if "results" in results:
            for task_name, task_results in results["results"].items():
                if isinstance(task_results, dict):
                    metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                            for k, v in task_results.items() if not k.endswith(",stderr")])
                    task_logger.info(f"  {task_name}: {metrics_str}")
        
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

            save_task_to_file(task_id, tasks_db[task_id])

    except Exception as e:
        import traceback
        
        # 记录错误日志
        task_logger.error(f"评测任务失败: {str(e)}")
        task_logger.error(f"错误详情:\n{traceback.format_exc()}")
        
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
            # 查找可用的生成式替代任务
            alternative_tasks = []
            for task in request.tasks:
                alt = get_generative_alternative(task)
                if alt:
                    alternative_tasks.append(f"  • {task} → {alt}")
            
            alt_section = ""
            if alternative_tasks:
                alt_section = "\n可用的生成式替代任务：\n" + "\n".join(alternative_tasks) + "\n"
            
            error_message = (
                "错误：当前模型不支持 Loglikelihood 类型的任务。\n\n"
                "原因：该模型的 API 不提供 prompt logprobs，无法运行需要 loglikelihood 的任务。\n\n"
                "解决方案：\n"
                "1. 【推荐】使用本地推理服务（如 Ollama、vLLM）：系统会自动切换到 local-completions 模型类型，完全支持 loglikelihood\n"
                "2. 选择生成式任务变体（使用正则匹配而非概率计算）\n"
                f"{alt_section}"
                "3. 使用 HuggingFace 模型（需要 GPU）\n\n"
                "技术说明：local-completions 通过 echo=True + logprobs 参数获取完整输入的 token 概率"
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
                tm = get_task_manager()
                available_tasks = set(tm.all_subtasks)
                available_groups = set(tm.all_groups)
                available_all = available_tasks.union(available_groups)
                if hasattr(tm, 'all_tasks'):
                    available_all = available_all.union(set(tm.all_tasks))
                
                if task_name not in available_all:
                    error_message = f"任务 '{task_name}' 未找到。请确认任务名称是否正确，或使用 '修复任务名称' 功能修复。"
                else:
                    error_message = f"任务 '{task_name}' 执行失败。请检查任务配置、模型参数或数据集是否正确。"
        
        with tasks_lock:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error_message"] = error_message
            tasks_db[task_id]["updated_at"] = datetime.now().isoformat()
            save_task_to_file(task_id, tasks_db[task_id])
    
    finally:
        # 恢复标准输出和错误输出
        if 'original_stdout' in locals():
            sys.stdout = original_stdout
        if 'original_stderr' in locals():
            sys.stderr = original_stderr


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
        
        # 获取后端类型和任务需求，确定 lm_eval model type
        backend_type = model_data.get("backend_type", "openai-api")
        from api.model_utils import determine_api_type
        lm_eval_model_type = determine_api_type(backend_type, request.tasks)
        
        # 使用模型数据构建 model_args，传入确定的 api_type
        final_model_args = get_model_args(model_data, api_type=lm_eval_model_type)
        # 如果请求中也提供了 model_args，合并它们（请求中的优先级更高）
        if request.model_args:
            final_model_args.update(request.model_args)
        
        # 设置 model 类型为自动确定的 lm_eval model type
        request.model = lm_eval_model_type
    
    # 验证 model_args 是否存在
    if not final_model_args:
        raise HTTPException(status_code=400, detail="model_args 是必需的，请提供 model_id 或 model_args")
    
    # 使用数据集中的 task_name（正确的任务名称）
    # lm-eval 会自动处理 group 类型的任务，评测其下所有子任务
    if request.datasets and len(request.datasets) > 0:
        # 从数据集信息中提取 task_name
        task_names_from_datasets = [ds.task_name for ds in request.datasets if ds.task_name]
        
        if task_names_from_datasets:
            normalized_tasks = task_names_from_datasets
        else:
            normalized_tasks = request.tasks
    else:
        normalized_tasks = request.tasks
    
    # 检查是否有无效的任务名称（使用异步版本避免阻塞事件循环）
    tm = await get_task_manager_async()
    available_tasks = set(tm.all_subtasks)
    available_groups = set(tm.all_groups)
    available_all = available_tasks.union(available_groups)
    
    # 尝试将 task_manager.all_tasks 也加入，以防遗漏
    if hasattr(tm, 'all_tasks'):
        available_all = available_all.union(set(tm.all_tasks))
    
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

    # 在后台运行评测（使用多进程包装器）
    # background_tasks.add_task(run_evaluation, task_id, request_with_normalized_tasks)
    # 直接启动进程，不使用 background_tasks (因为它使用线程池)
    run_evaluation_process_wrapper(task_id, request_with_normalized_tasks)

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
        # background_tasks.add_task(run_evaluation, task_id, original_request)
        run_evaluation_process_wrapper(task_id, original_request)
        
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
        # 总是尝试从文件重新加载以获取最新状态（因为子进程更新了文件，但不会更新此进程的内存）
        task_data = load_task_from_file(task_id)
        if task_data:
            # 只有当文件中的 updated_at 更新时才更新内存
            current_task = tasks_db.get(task_id)
            if not current_task or task_data.get("updated_at", "") >= current_task.get("updated_at", ""):
                 tasks_db[task_id] = task_data
        
        if task_id not in tasks_db:
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
        
        # 构建响应数据
        response_data = task.copy()
        
        # 如果任务已完成，尝试加载完整结果（但不包括原始样本，以免过大）
        if task["status"] == "completed" and task.get("result_file") and os.path.exists(task["result_file"]):
            try:
                with open(task["result_file"], "r", encoding="utf-8") as f:
                    full_results = json.load(f)
                    # 仅保留关键信息，移除可能很大的字段（如 samples）
                    # 但保留 results, config, versions, n-shot, git_hash 等
                    if "samples" in full_results:
                         # 采样少量样本用于展示（例如每个任务前5个）
                         samples_preview = {}
                         for task_name, samples in full_results["samples"].items():
                             samples_preview[task_name] = samples[:5] if isinstance(samples, list) else samples
                         full_results["samples_preview"] = samples_preview
                         del full_results["samples"] # 移除完整样本
                    
                    response_data["results"] = full_results
            except Exception as e:
                print(f"加载结果文件失败: {e}")
        
        return TaskResponse(**response_data)


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
             # Try load from file first
             task_data = load_task_from_file(task_id)
             if task_data:
                 tasks_db[task_id] = task_data
             else:
                 raise HTTPException(status_code=404, detail="任务不存在")
        
        # Reload checks
        task_data = load_task_from_file(task_id)
        if task_data:
             tasks_db[task_id] = task_data
        
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
        tm = await get_task_manager_async()
        return {
            "subtasks": tm.all_subtasks,
            "groups": tm.all_groups,
            "tags": tm.all_tags,
            "all_tasks": tm.all_tasks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.get("/{task_id}/logs")
async def get_task_logs(
    task_id: str, 
    lines: int = None,
    tail: bool = False
):
    """获取任务执行日志
    
    Args:
        task_id: 任务 ID
        lines: 返回的行数限制（默认返回全部）
        tail: 是否只返回最后几行（类似 tail 命令）
    """
    from pathlib import Path
    
    # 日志文件路径
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / "logs" / f"task_{task_id}.log"
    
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="日志文件不存在")
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        if lines is not None and lines > 0:
            all_lines = content.split("\n")
            if tail:
                # 返回最后 N 行
                selected_lines = all_lines[-lines:]
            else:
                # 返回前 N 行
                selected_lines = all_lines[:lines]
            content = "\n".join(selected_lines)
        
        return {
            "task_id": task_id,
            "log_file": str(log_file),
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {str(e)}")


@router.get("/{task_id}/logs/download")
async def download_task_logs(task_id: str):
    """下载任务日志文件"""
    from pathlib import Path
    from fastapi.responses import FileResponse
    
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / "logs" / f"task_{task_id}.log"
    
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="日志文件不存在")
    
    return FileResponse(
        str(log_file),
        media_type="text/plain",
        filename=f"task_{task_id}.log"
    )

