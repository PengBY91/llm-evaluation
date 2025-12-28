"""
数据集管理 API
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
from datasets import load_dataset, load_from_disk
import os
import threading
import json

router = APIRouter()

# 数据集存储目录
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# 数据集元数据存储目录
DATASETS_METADATA_DIR = Path(__file__).parent.parent.parent / "data" / "datasets_metadata"
DATASETS_METADATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据集索引文件
INDEX_FILE = DATA_DIR / "datasets_index.json"

# 数据集缓存（增加过期时间）
_datasets_cache: Optional[List[Dict[str, Any]]] = None
_cache_lock = threading.Lock()
_cache_timestamp: Optional[float] = None  # 缓存时间戳
CACHE_EXPIRE_SECONDS = 300  # 5分钟过期

# 数据集样本缓存（增加大小限制）
_samples_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_samples_cache_lock = threading.Lock()
MAX_SAMPLES_CACHE_SIZE = 100  # 最多缓存100个数据集的样本


import hashlib

def generate_dataset_id(path: str, config_name: Optional[str] = None) -> str:
    """生成数据集唯一 ID"""
    key = f"{path}:{config_name}" if config_name else path
    return hashlib.md5(key.encode("utf-8")).hexdigest()

class DatasetInfo(BaseModel):
    """数据集信息"""
    id: str
    name: str
    path: str
    config_name: Optional[str] = None
    task_name: Optional[str] = None
    description: Optional[str] = None
    local_path: Optional[str] = None
    is_local: bool = False
    splits: Optional[List[str]] = None
    num_examples: Optional[Dict[str, int]] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    output_type: Optional[str] = None
    subtasks: Optional[List[str]] = None


class DatasetAddRequest(BaseModel):
    """添加数据集请求"""
    dataset_path: str
    dataset_name: Optional[str] = None
    description: Optional[str] = None
    save_local: bool = True


class DatasetResponse(BaseModel):
    """数据集响应"""
    id: str
    name: str  # 数据集名称（文件夹名称，用于显示）
    path: str  # 数据集路径（HuggingFace 路径）
    config_name: Optional[str] = None  # 配置名称（子文件夹名称）
    task_name: Optional[str] = None  # 任务名称（从 TaskManager 获取，用于评测）
    description: Optional[str] = None
    local_path: Optional[str] = None
    is_local: bool
    splits: Optional[List[str]] = None
    num_examples: Optional[Dict[str, int]] = None
    category: Optional[str] = None  # 数据集类别
    tags: Optional[List[str]] = None
    output_type: Optional[str] = None
    subtasks: Optional[List[str]] = None


class DatasetListResponse(BaseModel):
    """数据集列表响应（带分页）"""
    datasets: List[DatasetResponse]
    total: int
    page: int
    page_size: int
    categories: List[str]  # 所有可用类别


def get_local_dataset_path(dataset_path: str, dataset_name: Optional[str] = None) -> Path:
    """获取本地数据集路径"""
    save_path = DATA_DIR / dataset_path.replace("/", "_")
    if dataset_name:
        save_path = save_path / dataset_name
    return save_path


def get_dataset_metadata_file_path(dataset_path: str, config_name: Optional[str] = None) -> Path:
    """获取数据集元数据文件路径"""
    # 使用路径和配置名称生成唯一的文件名
    file_name = dataset_path.replace("/", "_")
    if config_name:
        file_name = f"{file_name}_{config_name}"
    file_name = file_name.replace(":", "_").replace("*", "_")  # 替换特殊字符
    return DATASETS_METADATA_DIR / f"{file_name}.json"


def save_dataset_metadata(dataset_path: str, config_name: Optional[str] = None, metadata: Dict[str, Any] = None):
    """保存数据集元数据到文件"""
    if metadata is None:
        return
    
    metadata_file = get_dataset_metadata_file_path(dataset_path, config_name)
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存数据集元数据失败 {dataset_path}/{config_name}: {e}")


def load_dataset_metadata(dataset_path: str, config_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """从文件加载数据集元数据"""
    metadata_file = get_dataset_metadata_file_path(dataset_path, config_name)
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载数据集元数据失败 {dataset_path}/{config_name}: {e}")
    return None


def load_all_datasets_metadata() -> Dict[str, Dict[str, Any]]:
    """从文件加载所有数据集元数据"""
    metadata_dict = {}
    for metadata_file in DATASETS_METADATA_DIR.glob("*.json"):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                # 使用 path 和 config_name 作为键
                path = metadata.get("path", "")
                config_name = metadata.get("config_name")
                key = f"{path}:{config_name}" if config_name else path
                metadata_dict[key] = metadata
        except Exception as e:
            print(f"加载数据集元数据文件失败 {metadata_file}: {e}")
    return metadata_dict


def delete_dataset_metadata(dataset_path: str, config_name: Optional[str] = None):
    """删除数据集元数据文件"""
    metadata_file = get_dataset_metadata_file_path(dataset_path, config_name)
    if metadata_file.exists():
        try:
            metadata_file.unlink()
        except Exception as e:
            print(f"删除数据集元数据文件失败 {dataset_path}/{config_name}: {e}")


def infer_category(dataset_path: str, task_name: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
    """推断数据集类别"""
    # 从路径推断
    path_lower = dataset_path.lower()
    if any(x in path_lower for x in ['math', 'gsm8k', 'hendrycks']):
        return '数学推理'
    elif any(x in path_lower for x in ['arc', 'sciq', 'openbookqa']):
        return '科学问答'
    elif any(x in path_lower for x in ['hellaswag', 'winogrande', 'piqa']):
        return '常识推理'
    elif any(x in path_lower for x in ['mmlu', 'truthful']):
        return '多任务理解'
    elif any(x in path_lower for x in ['lambada', 'wikitext']):
        return '语言建模'
    elif any(x in path_lower for x in ['trivia', 'boolq']):
        return '问答'
    
    # 从标签推断
    if tags:
        tag_str = ' '.join(tags).lower()
        if any(x in tag_str for x in ['math', 'reasoning']):
            return '数学推理'
        elif any(x in tag_str for x in ['science', 'knowledge']):
            return '科学问答'
        elif any(x in tag_str for x in ['commonsense']):
            return '常识推理'
        elif any(x in tag_str for x in ['language']):
            return '语言建模'
    
    return '其他'


def rebuild_dataset_index() -> List[Dict[str, Any]]:
    """重建数据集索引（扫描磁盘并保存到文件）
    
    扫描逻辑：
    1. 对于直接包含 dataset_dict.json 的目录，作为单一数据集
    2. 对于包含子数据集的目录（如 arc/ARC-Challenge），同时列出父目录和子目录
       - 父目录标记为 "group"，选择它会评测所有子任务
       - 子目录标记为具体的数据集，可单独选择
    """
    global _datasets_cache
    
    try:
        # 加载已保存的数据集元数据
        saved_metadata = load_all_datasets_metadata()
        
        datasets = []
        
        # 1. 扫描本地 data 目录
        if not DATA_DIR.exists():
            print(f"数据目录不存在: {DATA_DIR}")
            _datasets_cache = []
            return []
        
        # 获取所有第一层子目录
        try:
            top_level_dirs = [d for d in DATA_DIR.iterdir() 
                             if d.is_dir() 
                             and not d.name.startswith('.') 
                             and d.name not in ['datasets_metadata', 'tasks']]
        except Exception as e:
            print(f"读取数据目录失败: {e}")
            _datasets_cache = []
            return []
        
        # 延迟加载 TaskManager（仅在需要时加载）
        task_manager = None
        
        for dir_path in top_level_dirs:
            try:
                dataset_name = dir_path.name
                
                # 检查是否是直接数据集（直接包含 dataset_dict.json）
                is_direct_dataset = (
                    (dir_path / 'dataset_dict.json').exists() or 
                    (dir_path / 'dataset_info.json').exists()
                )
                
                # 检查子目录
                sub_datasets = []
                if not is_direct_dataset:
                    try:
                        for sub_dir in dir_path.iterdir():
                            if sub_dir.is_dir():
                                if ((sub_dir / 'dataset_dict.json').exists() or 
                                    (sub_dir / 'dataset_info.json').exists()):
                                    sub_datasets.append(sub_dir)
                    except (PermissionError, OSError) as e:
                        print(f"无法访问子目录 {dir_path}: {e}")
                        continue
                
                if not is_direct_dataset and not sub_datasets:
                    # 跳过非数据集目录
                    continue
                
                # 延迟加载 TaskManager
                if task_manager is None:
                    try:
                        from lm_eval.tasks import TaskManager
                        task_manager = TaskManager()
                    except Exception as e:
                        print(f"初始化 TaskManager 失败: {e}")
                
                # 创建父目录条目（对于有子数据集的情况，这代表整个任务组）
                is_group = len(sub_datasets) > 0
                parent_info = {
                    "id": dataset_name,
                    "name": dataset_name,
                    "path": dataset_name,
                    "config_name": None,
                    "task_name": dataset_name,
                    "description": f"Task: {dataset_name}" + (f"（包含 {len(sub_datasets)} 个子任务）" if is_group else ""),
                    "local_path": str(dir_path),
                    "is_local": True,
                    "splits": None,
                    "num_examples": None,
                    "category": infer_category(dataset_name),
                    "tags": ["group"] if is_group else [],
                    "output_type": None,
                    "subtasks": [sub.name for sub in sub_datasets] if sub_datasets else None
                }
                
                # 如果是直接数据集，尝试加载 split 信息
                if is_direct_dataset:
                    try:
                        dataset = load_from_disk(str(dir_path))
                        splits = list(dataset.keys())
                        parent_info["splits"] = splits
                        parent_info["num_examples"] = {split: len(dataset[split]) for split in splits}
                    except Exception as e:
                        print(f"扫描数据集 {dataset_name} 的 splits 失败: {e}")
                elif is_group and sub_datasets:
                    # 如果是组，且自己没有 splits，尝试从第一个子任务获取 splits 以供预览
                    try:
                        first_sub = sub_datasets[0]
                        dataset = load_from_disk(str(first_sub))
                        splits = list(dataset.keys())
                        parent_info["splits"] = splits
                        parent_info["num_examples"] = {split: len(dataset[split]) for split in splits}
                    except Exception as e:
                        print(f"扫描数据集组 {dataset_name} 的第一个子任务 splits 失败: {e}")
                
                # 验证任务名称
                if task_manager is not None:
                    try:
                        if dataset_name in task_manager.all_groups:
                            parent_info["tags"].append("lm_eval_group")
                        elif dataset_name in task_manager.all_subtasks:
                            parent_info["tags"].append("lm_eval_task")
                    except Exception as e:
                        print(f"验证任务名称失败 {dataset_name}: {e}")
                
                # 合并已保存的元数据
                if dataset_name in saved_metadata:
                    try:
                        saved_meta = saved_metadata[dataset_name]
                        # 合并所有常用字段
                        for field in ["description", "category", "task_name", "output_type", "splits", "num_examples"]:
                            if saved_meta.get(field) is not None:
                                parent_info[field] = saved_meta[field]
                        
                        if "tags" in saved_meta and isinstance(saved_meta["tags"], list):
                            parent_info["tags"] = list(set(parent_info["tags"] + saved_meta["tags"]))
                    except Exception as e:
                        print(f"合并元数据失败 {dataset_name}: {e}")
                
                datasets.append(parent_info)
                
                # 子任务信息已经包含在父级的 subtasks 字段中
                # 不再为每个子任务创建单独的索引条目
                # lm-eval 会自动处理 group 下的子任务
                
            except Exception as e:
                print(f"处理数据集目录失败 {dir_path}: {e}")
                # 继续处理其他数据集
                continue
        
        # 排序：有 task_name 的排前面，然后按名称排
        try:
            datasets.sort(key=lambda x: (x["task_name"] is None, x["name"]))
        except Exception as e:
            print(f"排序数据集失败: {e}")
        
        # 保存到索引文件
        try:
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(datasets, f, ensure_ascii=False, indent=2)
            print(f"数据集索引已保存到: {INDEX_FILE}，共 {len(datasets)} 个数据集")
        except Exception as e:
            print(f"保存数据集索引失败: {e}")
        
        _datasets_cache = datasets
        return datasets
        
    except Exception as e:
        print(f"重建数据集索引失败: {e}")
        import traceback
        traceback.print_exc()
        _datasets_cache = []
        return []


def load_all_datasets() -> List[Dict[str, Any]]:
    """加载所有数据集（带缓存、过期和错误恢复）"""
    global _datasets_cache, _cache_timestamp
    
    import time
    current_time = time.time()
    
    with _cache_lock:
        # 检查缓存是否存在且未过期
        if (_datasets_cache is not None and 
            _cache_timestamp is not None and 
            (current_time - _cache_timestamp) < CACHE_EXPIRE_SECONDS):
            return _datasets_cache
        
        # 缓存已过期或不存在
        datasets = None
        
        # 尝试从索引文件加载
        if INDEX_FILE.exists():
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    datasets = json.load(f)
                
                # 验证数据格式
                if not isinstance(datasets, list):
                    print(f"索引文件格式错误，预期列表但得到 {type(datasets)}")
                    datasets = None
                else:
                    print(f"从缓存加载 {len(datasets)} 个数据集")
                    
            except json.JSONDecodeError as e:
                print(f"解析数据集索引文件失败: {e}，将重新扫描")
                datasets = None
            except Exception as e:
                print(f"读取数据集索引文件失败: {e}，将重新扫描")
                datasets = None
        
        # 如果没有索引文件或读取失败，重建索引
        if datasets is None:
            print("重建数据集索引...")
            try:
                datasets = rebuild_dataset_index()
            except Exception as e:
                print(f"重建索引失败: {e}")
                _datasets_cache = []
                return []
        
        # 检查并修复缺失的 ID (兼容旧索引文件)
        needs_save = False
        try:
            for dataset in datasets:
                if not isinstance(dataset, dict):
                    print(f"跳过无效的数据集条目: {dataset}")
                    continue
                    
                if dataset.get("id") is None:
                    dataset["id"] = generate_dataset_id(
                        dataset.get("path", ""), 
                        dataset.get("config_name")
                    )
                    needs_save = True
        except Exception as e:
            print(f"修复数据集 ID 失败: {e}")
        
        # 如果有更新，保存回文件
        if needs_save:
            try:
                with open(INDEX_FILE, "w", encoding="utf-8") as f:
                    json.dump(datasets, f, ensure_ascii=False, indent=2)
                print(f"数据集索引已更新并保存到: {INDEX_FILE}")
            except Exception as e:
                print(f"保存更新后的数据集索引失败: {e}")
        
        _datasets_cache = datasets
        _cache_timestamp = current_time
        return datasets


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    category: Optional[str] = Query(None, description="按类别过滤"),
    is_local: Optional[bool] = Query(None, description="是否只显示本地数据集"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=500, description="每页数量（最大500）"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    groups_only: bool = Query(False, description="是否只返回 Group 级别的数据集（不包含子任务）")
):
    """获取数据集列表（支持分类过滤和分页）
    
    groups_only=True 时只返回 Group 级别的数据集，用于数据集管理页面
    groups_only=False 时返回所有数据集（包括子任务），用于任务创建
    """
    all_datasets = load_all_datasets()
    
    # 过滤
    filtered = []
    
    for d in all_datasets:
        # 如果只要 Group，跳过子任务
        if groups_only and d.get("parent"):
            continue
        
        # 类别过滤
        if category and d.get("category") != category:
            continue
        
        # 本地状态过滤
        if is_local is not None and d.get("is_local") != is_local:
            continue
        
        # 搜索过滤
        if search:
            search_lower = search.lower()
            matches = (
                search_lower in d.get("name", "").lower() or
                search_lower in d.get("path", "").lower() or
                search_lower in d.get("description", "").lower() or
                search_lower in d.get("task_name", "").lower() or
                (d.get("config_name") and search_lower in d.get("config_name", "").lower())
            )
            if not matches:
                continue
        
        filtered.append(d)
    
    # 兼容旧代码：如果搜索结果是子任务，确保父 group 也被包含（仅在非 groups_only 模式）
    if search and not groups_only:
        matched_parents = set(d.get("parent") for d in filtered if d.get("parent"))
        for d in all_datasets:
            if d.get("id") in matched_parents and d not in filtered:
                # 插入到开头
                filtered.insert(0, d)
    
    # 排序：Group 在前，子任务紧随其后
    def sort_key(d):
        if d.get("parent"):
            # 子任务排在其父 group 后面
            return (d.get("parent"), 1, d.get("name", ""))
        else:
            # Group 按名称排序
            return (d.get("id"), 0, "")
    
    filtered.sort(key=sort_key)
    
    # 获取所有类别
    categories = sorted(set(d.get("category", "其他") for d in all_datasets if d.get("category")))
    
    # 分页 - 按 group 计算页面，确保 group 和其子任务在同一页
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]
    
    return DatasetListResponse(
        datasets=[DatasetResponse(**d) for d in paginated],
        total=total,
        page=page,
        page_size=page_size,
        categories=categories
    )


@router.post("/refresh-cache")
async def refresh_cache():
    """刷新数据集缓存（触发重新扫描并更新索引）"""
    global _datasets_cache, _cache_timestamp
    
    try:
        # 清除缓存
        with _cache_lock:
            _datasets_cache = None
            _cache_timestamp = None
        
        # 异步重建索引，避免阻塞
        import asyncio
        loop = asyncio.get_event_loop()
        
        # 使用超时保护，防止长时间阻塞
        try:
            datasets = await asyncio.wait_for(
                loop.run_in_executor(None, rebuild_dataset_index),
                timeout=30.0  # 30秒超时
            )
            return {
                "message": f"缓存已刷新，数据集索引已更新。共找到 {len(datasets)} 个数据集",
                "count": len(datasets)
            }
        except asyncio.TimeoutError:
            return {
                "message": "缓存刷新超时，但后台仍在继续处理。请稍后刷新页面查看结果。",
                "warning": True
            }
            
    except Exception as e:
        import traceback
        error_msg = f"缓存刷新失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {
            "message": error_msg,
            "error": True
        }


@router.get("/samples")
async def get_dataset_samples(
    dataset_name: str = Query(..., description="数据集名称"), 
    split: str = Query("train"), 
    limit: int = Query(2, le=100)  # 限制最大100条
):
    """获取数据集样本（带缓存和性能优化）"""
    try:
        # URL 解码数据集名称（处理可能的编码问题）
        import urllib.parse
        dataset_name = urllib.parse.unquote(dataset_name)
        
        cache_key = f"{dataset_name}:{split}"
        
        # 检查缓存
        with _samples_cache_lock:
            if cache_key in _samples_cache and len(_samples_cache[cache_key]) >= limit:
                return _samples_cache[cache_key][:limit]
        
        # 从缓存的数据集列表中查找数据集信息
        all_datasets = load_all_datasets()
        dataset_info = None
        
        # 首先尝试精确匹配 name
        for d in all_datasets:
            if d.get("name") == dataset_name:
                dataset_info = d
                break
        
        # 如果找不到，尝试多种匹配方式
        if not dataset_info:
            # 将数据集名称中的斜杠转换为下划线，用于匹配目录名格式
            dataset_name_normalized = dataset_name.replace("/", "_")
            # 也尝试将下划线转换为斜杠
            dataset_name_with_slash = dataset_name.replace("_", "/")
        
        # 尝试解析数据集名称，可能是 path/config_name 格式
        # 例如: EleutherAI/hendrycks/math_intermediate_algebra
        # 可能是: path = EleutherAI/hendrycks/math, config_name = intermediate_algebra
        # 或者: path = EleutherAI/hendrycks_math, config_name = intermediate_algebra
        dataset_parts = dataset_name.split("/")
        possible_paths = []
        if len(dataset_parts) > 1:
            # 尝试不同的路径组合
            # 例如: EleutherAI/hendrycks/math_intermediate_algebra
            # 可能是: EleutherAI/hendrycks/math + intermediate_algebra
            # 或者: EleutherAI/hendrycks_math + intermediate_algebra
            last_part = dataset_parts[-1]
            if "_" in last_part:
                # 最后一部分可能包含 config_name
                parts = last_part.split("_", 1)
                if len(parts) == 2:
                    possible_paths.append(("/".join(dataset_parts[:-1]) + "/" + parts[0], parts[1]))
                    possible_paths.append(("/".join(dataset_parts[:-1]) + "_" + parts[0], parts[1]))
        
        for d in all_datasets:
            path = d.get("path", "")
            config_name = d.get("config_name")
            name = d.get("name", "")
            
            # 尝试匹配 path_config_name 格式
            if config_name:
                expected_name = f"{path}_{config_name}"
                # 也尝试路径中的斜杠替换为下划线的格式
                path_normalized = path.replace("/", "_")
                expected_name_normalized = f"{path_normalized}_{config_name}"
            else:
                expected_name = path
                path_normalized = path.replace("/", "_")
                expected_name_normalized = path_normalized
            
            # 多种匹配方式：支持斜杠和下划线的互相转换
            matches = (
                expected_name == dataset_name or 
                expected_name_normalized == dataset_name or
                expected_name_normalized == dataset_name_normalized or
                expected_name == dataset_name_normalized or
                expected_name == dataset_name_with_slash or
                path == dataset_name or
                path == dataset_name_normalized or
                path == dataset_name_with_slash or
                path_normalized == dataset_name or
                path_normalized == dataset_name_normalized or
                name == dataset_name or
                name == dataset_name_normalized or
                name == dataset_name_with_slash or
                name.replace("/", "_") == dataset_name_normalized or
                name.replace("_", "/") == dataset_name or
                name.replace("_", "/") == dataset_name_with_slash
            )
            
            # 检查可能的路径组合
            if not matches and possible_paths:
                for possible_path, possible_config in possible_paths:
                    if path == possible_path and config_name == possible_config:
                        matches = True
                        break
                    path_norm = possible_path.replace("/", "_")
                    if path_normalized == path_norm and config_name == possible_config:
                        matches = True
                        break
            
            # 最后尝试：直接比较路径的各个部分（忽略分隔符）
            if not matches:
                # 将路径和名称都标准化（统一使用下划线）
                dataset_name_std = dataset_name.replace("/", "_").replace("-", "_")
                path_std = path.replace("/", "_").replace("-", "_")
                name_std = name.replace("/", "_").replace("-", "_")
                
                # 如果标准化后的名称或路径完全匹配
                if (dataset_name_std == path_std or 
                    dataset_name_std == name_std or
                    (config_name and dataset_name_std == f"{path_std}_{config_name}")):
                    matches = True
            
            if matches:
                dataset_info = d
                break
        
        if not dataset_info:
            # 提供简洁的错误信息，避免过多输出
            available_names = [d.get("name", "unknown") for d in all_datasets[:10]]
            error_msg = f"数据集不存在: {dataset_name}。可用数据集（前10个）: {', '.join(available_names)}"
            
            raise HTTPException(
                status_code=404, 
                detail=error_msg
            )
        
        # 检查是否有本地路径
        local_path = dataset_info.get("local_path")
        if not local_path:
            raise HTTPException(
                status_code=404, 
                detail=f"数据集 '{dataset_name}' 没有本地路径，无法加载样本。请先下载数据集。"
            )
        
        # 检查本地路径是否存在
        from pathlib import Path
        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            raise HTTPException(
                status_code=404,
                detail=f"数据集本地路径不存在: {local_path}"
            )
        
        # 从本地加载数据集（使用超时保护）
        try:
            dataset_path_str = str(local_path)
            dataset = load_from_disk(dataset_path_str)
            
            # 验证数据集是否成功加载
            if not dataset or not isinstance(dataset, dict):
                raise ValueError(f"数据集加载失败：返回的数据不是字典格式")
                
        except Exception as e:
            error_detail = f"加载本地数据集失败: {str(e)}"
            raise HTTPException(status_code=500, detail=error_detail)
        
        # 检查 split 是否存在
        if split not in dataset:
            available_splits = list(dataset.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Split '{split}' 不存在。可用 splits: {', '.join(available_splits)}"
            )
        
        # 获取样本
        try:
            split_dataset = dataset[split]
            samples = split_dataset[:limit]
            
            # 转换为字典列表
            if isinstance(samples, dict):
                # 单个样本
                sample_list = [{k: v[i] for k, v in samples.items()} for i in range(len(next(iter(samples.values()))))]
            else:
                sample_list = [dict(sample) for sample in samples]
                
        except Exception as e:
            error_detail = f"读取样本失败: {str(e)}"
            raise HTTPException(status_code=500, detail=error_detail)
        
        # 更新缓存（带大小限制）
        with _samples_cache_lock:
            # 如果缓存已满，删除最旧的条目
            if len(_samples_cache) >= MAX_SAMPLES_CACHE_SIZE:
                # 删除第一个（最旧的）条目
                first_key = next(iter(_samples_cache))
                del _samples_cache[first_key]
            
            _samples_cache[cache_key] = sample_list
        
        return sample_list
        
    except HTTPException:
        raise
    except Exception as e:
        # 捕获所有未预期的异常
        import traceback
        error_detail = f"获取数据集样本时发生错误: {str(e)}"
        print(error_detail)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_detail)




@router.get("/readme/{dataset_id:path}")
async def get_dataset_readme(dataset_id: str):
    """获取数据集 README 内容
    
    使用 :path 类型以支持包含 '/' 的 ID（如 arc/ARC-Challenge）
    """
    from urllib.parse import unquote
    dataset_id = unquote(dataset_id)
    
    all_datasets = load_all_datasets()
    dataset_info = None
    
    # 1. 优先匹配 ID
    for ds in all_datasets:
        if ds.get("id") == dataset_id:
            dataset_info = ds
            break
            
    # 2. 兼容匹配 name
    if not dataset_info:
        for ds in all_datasets:
            if ds.get("name") == dataset_id:
                dataset_info = ds
                break
    
    if not dataset_info:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    # 获取任务名称（或文件夹名）
    task_name = dataset_info.get("task_name") or dataset_info.get("name")
    if not task_name:
        return {"content": None}

    # 尝试在 lm_eval/tasks 目录下查找对应的 README.md
    # 路径通常是 lm_eval/tasks/{task_name}/README.md
    # 或者 lm_eval/tasks/{group_name}/{subtask_name}/README.md (复杂情况暂不处理，先处理顶层)
    
    lm_eval_tasks_dir = Path(__file__).parent.parent.parent / "lm_eval" / "tasks"
    
    # 尝试直接匹配任务名目录
    readme_path = lm_eval_tasks_dir / task_name / "README.md"
    
    # 如果找不到，尝试处理包含子配置的情况 (e.g. bbh/boolean_expressions -> bbh)
    if not readme_path.exists():
        # 如果 task_name 包含下划线或斜杠，尝试取第一部分
        # e.g. mmlu_abstract_algebra -> mmlu
        # e.g. bbh/boolean_expressions -> bbh
        
        parts = task_name.replace("/", "_").split("_")
        if len(parts) > 1:
            # 尝试逐级查找
            # 比如对于 mmlu_abstract_algebra，先找 mmlu
            potential_group = parts[0]
            group_readme = lm_eval_tasks_dir / potential_group / "README.md"
            if group_readme.exists():
                readme_path = group_readme

    if readme_path.exists():
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"content": content}
        except Exception as e:
            print(f"读取 README 失败: {e}")
            return {"content": None}
    
    return {"content": None}


@router.post("/", response_model=DatasetResponse)
async def add_dataset(request: DatasetAddRequest):
    """添加/下载数据集"""
    global _datasets_cache
    
    try:
        # 加载数据集
        if request.dataset_name:
            dataset = load_dataset(
                request.dataset_path,
                request.dataset_name,
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                request.dataset_path,
                trust_remote_code=True
            )
        
        local_path = None
        
        # 如果要求保存到本地
        if request.save_local:
            save_path = get_local_dataset_path(request.dataset_path, request.dataset_name)
            save_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(save_path))
            local_path = str(save_path)
        
        splits = list(dataset.keys())
        num_examples = {split: len(dataset[split]) for split in splits}
        
        # 尝试从 TaskManager 获取正确的任务名称
        task_name = None
        try:
            from lm_eval.tasks import TaskManager
            task_manager = TaskManager()
            for t_name in task_manager.all_subtasks:
                task_info = task_manager.task_index.get(t_name, {})
                yaml_path = task_info.get("yaml_path", -1)
                if yaml_path != -1:
                    try:
                        from lm_eval import utils
                        config = utils.load_yaml_config(yaml_path, mode="simple")
                        if (config.get("dataset_path") == request.dataset_path and 
                            config.get("dataset_name") == request.dataset_name):
                            task_name = t_name
                            break
                    except Exception:
                        pass
        except Exception:
            pass
        
        # 如果没有找到匹配的任务名称，使用 path_config_name 格式作为后备
        if task_name is None:
            task_name = request.dataset_path + (f"_{request.dataset_name}" if request.dataset_name else "")
        
        # 数据集名称使用路径（将斜杠转换为下划线，用于显示）
        display_name = request.dataset_path.replace("/", "_")
        if request.dataset_name:
            display_name = f"{display_name}_{request.dataset_name}"
        
        # 构建数据集响应
        dataset_response = DatasetResponse(
            id=generate_dataset_id(request.dataset_path, request.dataset_name),
            name=display_name,  # 使用路径作为显示名称
            path=request.dataset_path,
            config_name=request.dataset_name,
            task_name=task_name,  # 从 TaskManager 获取的任务名称（用于评测）
            description=request.description,
            local_path=local_path,
            is_local=request.save_local,
            splits=splits,
            num_examples=num_examples,
            category=infer_category(request.dataset_path),
            tags=[]
        )
        
        # 保存数据集元数据到文件
        metadata = {
            "path": request.dataset_path,
            "config_name": request.dataset_name,
            "description": request.description,
            "category": dataset_response.category,
            "tags": dataset_response.tags,
            "task_name": task_name,
            "local_path": local_path,
            "is_local": request.save_local
        }
        save_dataset_metadata(request.dataset_path, request.dataset_name, metadata)
        
        # 清除缓存
        with _cache_lock:
            _datasets_cache = None
        
        return dataset_response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"添加数据集失败: {str(e)}")


@router.delete("/{dataset_id:path}")
async def delete_dataset(dataset_id: str):
    """删除本地数据集
    
    使用 :path 类型以支持包含 '/' 的 ID（如 arc/ARC-Challenge）
    """
    global _datasets_cache
    
    from urllib.parse import unquote
    dataset_id = unquote(dataset_id)
    
    # 先查找数据集信息，以便删除元数据
    all_datasets = load_all_datasets()
    dataset_info = None
    dataset_name = dataset_id  # 兼容旧代码
    
    # 优先通过 ID 匹配
    for ds in all_datasets:
        if ds.get("id") == dataset_id:
            dataset_info = ds
            dataset_name = ds.get("name", dataset_id)
            break
    
    # 兼容通过 name 匹配
    if not dataset_info:
        for ds in all_datasets:
            if ds.get("name") == dataset_id:
                dataset_info = ds
                dataset_name = ds.get("name", dataset_id)
                break
    
    dataset_path = DATA_DIR / dataset_name.replace("/", "_")
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="本地数据集不存在")
    
    import shutil
    try:
        shutil.rmtree(dataset_path)
        
        # 删除数据集元数据（如果存在）
        if dataset_info:
            delete_dataset_metadata(dataset_info.get("path"), dataset_info.get("config_name"))
        
        # 清除缓存
        with _cache_lock:
            _datasets_cache = None
        
        # 清除样本缓存
        with _samples_cache_lock:
            keys_to_remove = [k for k in _samples_cache.keys() if k.startswith(dataset_name)]
            for k in keys_to_remove:
                del _samples_cache[k]
        
        return {"message": "数据集已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/{dataset_id:path}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """获取单个数据集详情（支持 ID 或 Name）
    
    使用 :path 类型以支持包含 '/' 的 ID（如 arc/ARC-Challenge）
    """
    all_datasets = load_all_datasets()
    
    # URL 解码（前端可能会编码 /）
    from urllib.parse import unquote
    dataset_id = unquote(dataset_id)
    
    # 1. 优先匹配 ID（精确匹配）
    for dataset_info in all_datasets:
        if dataset_info.get("id") == dataset_id:
            return DatasetResponse(**dataset_info)
    
    for dataset_info in all_datasets:
        if dataset_info["name"] == dataset_id:
            return DatasetResponse(**dataset_info)
    
    # 3. 如果没找到，尝试从本地查找（兼容旧代码）
    # 注意：这里的 dataset_id 其实是 dataset_name
    dataset_name = dataset_id
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            if dataset_dir.name == dataset_name.replace("/", "_"):
                try:
                    dataset = load_from_disk(str(dataset_dir))
                    splits = list(dataset.keys())
                    num_examples = {split: len(dataset[split]) for split in splits}
                    
                    return DatasetResponse(
                        id=generate_dataset_id(dataset_name, None),
                        name=dataset_name,
                        path=dataset_name,
                        local_path=str(dataset_dir),
                        is_local=True,
                        splits=splits,
                        num_examples=num_examples
                    )
                except Exception:
                    pass
    
    # 尝试从 HuggingFace 加载信息
    try:
        from datasets import get_dataset_infos
        dataset_infos = get_dataset_infos(dataset_name)
        
        if dataset_infos:
            info = dataset_infos[list(dataset_infos.keys())[0]]
            return DatasetResponse(
                id=generate_dataset_id(dataset_name, None),
                name=dataset_name,
                path=dataset_name,
                is_local=False,
                description=info.description,
                splits=list(info.splits.keys()) if info.splits else None
            )
    except Exception:
        pass
    
    raise HTTPException(status_code=404, detail="数据集不存在")


