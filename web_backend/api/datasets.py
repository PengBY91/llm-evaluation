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

# 数据集缓存
_datasets_cache: Optional[List[Dict[str, Any]]] = None
_cache_lock = threading.Lock()

# 数据集样本缓存
_samples_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_samples_cache_lock = threading.Lock()


class DatasetInfo(BaseModel):
    """数据集信息"""
    name: str
    path: str
    config_name: Optional[str] = None
    description: Optional[str] = None
    local_path: Optional[str] = None
    is_local: bool = False
    splits: Optional[List[str]] = None
    num_examples: Optional[Dict[str, int]] = None


class DatasetAddRequest(BaseModel):
    """添加数据集请求"""
    dataset_path: str
    dataset_name: Optional[str] = None
    description: Optional[str] = None
    save_local: bool = True


class DatasetResponse(BaseModel):
    """数据集响应"""
    name: str  # 数据集名称（来自 YAML 配置中的 task 字段）
    path: str  # 数据集路径（HuggingFace 路径）
    config_name: Optional[str] = None  # 配置名称（子文件夹名称）
    task_name: Optional[str] = None  # 任务名称（从 TaskManager 获取，用于评测）
    description: Optional[str] = None
    local_path: Optional[str] = None
    is_local: bool
    splits: Optional[List[str]] = None
    num_examples: Optional[Dict[str, int]] = None
    category: Optional[str] = None  # 数据集类别
    tags: Optional[List[str]] = None  # 标签列表


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


def load_all_datasets() -> List[Dict[str, Any]]:
    """加载所有数据集（带缓存）"""
    global _datasets_cache
    
    with _cache_lock:
        # 如果缓存存在，直接返回（避免重复扫描）
        if _datasets_cache is not None:
            return _datasets_cache
        
        # 加载已保存的数据集元数据
        saved_metadata = load_all_datasets_metadata()
        
        datasets = []
        
        # 扫描本地 data 目录
        # 数据集名称将使用 YAML 配置中的 task 字段（在匹配到 TaskManager 后更新）
        if DATA_DIR.exists():
            for dataset_dir in DATA_DIR.iterdir():
                if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                    # 跳过 tasks 等特殊目录
                    if dataset_dir.name in ['tasks', 'README.md', 'USAGE.md']:
                        continue
                    
                    config_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    
                    if config_dirs:
                        # 有子文件夹，每个子文件夹是一个配置
                        for config_dir in config_dirs:
                            # 数据集路径：文件夹名称（保持原样，用于匹配 TaskManager）
                            dataset_path = dataset_dir.name  # 保持原样，不替换下划线
                            # 配置名称：子文件夹名称
                            config_name = config_dir.name
                            # 数据集名称：初始使用文件夹名称，后续将从 YAML 配置中的 task 字段更新
                            dataset_name = dataset_dir.name  # 初始值，后续会更新为 task 字段
                            
                            # 尝试加载数据集，即使失败也添加到列表中（稍后从 TaskManager 匹配 task_name 和 task 字段）
                            splits = None
                            num_examples = None
                            try:
                                dataset = load_from_disk(str(config_dir))
                                splits = list(dataset.keys())
                                num_examples = {split: len(dataset[split]) for split in splits}
                            except Exception:
                                # 数据集加载失败，但仍然添加到列表中（可能数据集文件损坏，但可以从 TaskManager 匹配 task_name）
                                pass
                            
                            # 创建数据集对象
                            dataset_info = {
                                "name": dataset_name,  # 初始值，后续将从 YAML 配置中的 task 字段更新
                                "path": dataset_path,  # 保持原样，用于匹配 TaskManager
                                "config_name": config_name,
                                "task_name": None,  # 稍后从 TaskManager 获取
                                "local_path": str(config_dir),
                                "is_local": True,
                                "splits": splits,
                                "num_examples": num_examples,
                                "category": infer_category(dataset_path),
                                "tags": []
                            }
                            
                            # 合并已保存的元数据（如果存在）
                            metadata_key = f"{dataset_path}:{config_name}"
                            if metadata_key in saved_metadata:
                                saved_meta = saved_metadata[metadata_key]
                                # 合并元数据，已保存的优先级更高
                                if "description" in saved_meta:
                                    dataset_info["description"] = saved_meta["description"]
                                if "category" in saved_meta:
                                    dataset_info["category"] = saved_meta["category"]
                                if "tags" in saved_meta:
                                    dataset_info["tags"] = saved_meta["tags"]
                            
                            datasets.append(dataset_info)
                    else:
                        # 没有子文件夹，直接是数据集
                        dataset_path = dataset_dir.name  # 保持原样，不替换下划线
                        dataset_name = dataset_dir.name  # 初始值，后续将从 YAML 配置中的 task 字段更新
                        
                        # 尝试加载数据集，即使失败也添加到列表中（稍后从 TaskManager 匹配 task_name 和 task 字段）
                        splits = None
                        num_examples = None
                        try:
                            dataset = load_from_disk(str(dataset_dir))
                            splits = list(dataset.keys())
                            num_examples = {split: len(dataset[split]) for split in splits}
                        except Exception:
                            # 数据集加载失败，但仍然添加到列表中（可能数据集文件损坏，但可以从 TaskManager 匹配 task_name）
                            pass
                        
                        # 创建数据集对象
                        dataset_info = {
                            "name": dataset_name,  # 初始值，后续将从 YAML 配置中的 task 字段更新
                            "path": dataset_path,  # 保持原样，用于匹配 TaskManager
                            "config_name": None,
                            "task_name": None,  # 稍后从 TaskManager 获取
                            "local_path": str(dataset_dir),
                            "is_local": True,
                            "splits": splits,
                            "num_examples": num_examples,
                            "category": infer_category(dataset_path),
                            "tags": []
                        }
                        
                        # 合并已保存的元数据（如果存在）
                        metadata_key = dataset_path
                        if metadata_key in saved_metadata:
                            saved_meta = saved_metadata[metadata_key]
                            # 合并元数据，已保存的优先级更高
                            if "description" in saved_meta:
                                dataset_info["description"] = saved_meta["description"]
                            if "category" in saved_meta:
                                dataset_info["category"] = saved_meta["category"]
                            if "tags" in saved_meta:
                                dataset_info["tags"] = saved_meta["tags"]
                        
                        datasets.append(dataset_info)
        
        # 从 TaskManager 获取所有可用任务（数据集）
        # 注意：这里需要先扫描本地目录，然后再从 TaskManager 匹配，确保所有本地数据集都能匹配到 task_name
        try:
            from lm_eval.tasks import TaskManager
            task_manager = TaskManager()
            
            # 对于 config_name 为 "all" 的数据集，先尝试匹配组（group）
            # 例如：cais_mmlu/all 应该匹配到 mmlu 组
            for dataset in datasets:
                if dataset.get("config_name") == "all" and dataset.get("task_name") is None:
                    dataset_path = dataset.get("path", "")
                    # 尝试从 dataset_path 推断组名（例如：cais_mmlu -> mmlu）
                    # 移除前缀（如 cais_）后，尝试匹配组名
                    possible_group_names = []
                    if "_" in dataset_path:
                        # 尝试移除前缀
                        parts = dataset_path.split("_")
                        for i in range(1, len(parts) + 1):
                            possible_group_names.append("_".join(parts[i:]))
                    else:
                        possible_group_names.append(dataset_path)
                    
                    # 检查是否有匹配的组
                    for group_name in task_manager.all_groups:
                        if group_name in possible_group_names:
                            dataset["task_name"] = group_name
                            break
            
            # 遍历 TaskManager 中的所有任务（包括独立任务和组），尝试匹配已扫描的本地数据集
            # 先处理独立任务（subtasks）
            for task_name in task_manager.all_subtasks:
                task_info = task_manager.task_index.get(task_name, {})
                yaml_path = task_info.get("yaml_path", -1)
                
                if yaml_path != -1:
                    try:
                        from lm_eval import utils
                        config = utils.load_yaml_config(yaml_path, mode="simple")
                        dataset_path = config.get("dataset_path")
                        
                        # 获取 task 字段作为数据集名称（对于独立任务，这是字符串）
                        task_name_from_config = config.get("task")
                        # 如果 task 是列表（不应该在独立任务中出现），取第一个
                        if isinstance(task_name_from_config, list):
                            task_name_from_config = task_name_from_config[0] if task_name_from_config else None
                        
                        if dataset_path:
                            dataset_name = config.get("dataset_name")
                            
                            # 检查是否已在列表中（匹配 path 和 config_name）
                            # 注意：TaskManager 中的 dataset_path 可能使用斜杠（如 cais/mmlu），
                            # 而文件夹名称使用下划线（如 cais_mmlu），需要同时匹配两种格式
                            existing_index = None
                            best_match_task_name = None  # 记录最佳匹配的任务名称（dataset_name 为 None 的优先）
                            for idx, d in enumerate(datasets):
                                # 匹配逻辑：path 和 config_name 都相同
                                # 注意：config_name 可能为 None，需要特殊处理
                                d_path = d.get("path")
                                d_config = d.get("config_name")
                                
                                # 匹配 path：支持斜杠和下划线的互相转换
                                # TaskManager 中的 dataset_path 可能使用斜杠，文件夹名称使用下划线
                                path_matches = (
                                    d_path == dataset_path or  # 直接匹配
                                    d_path == dataset_path.replace("/", "_") or  # TaskManager 的斜杠转下划线后匹配
                                    d_path.replace("_", "/") == dataset_path or  # 文件夹的下划线转斜杠后匹配
                                    d_path.replace("_", "/") == dataset_path.replace("/", "_")  # 都转换后匹配
                                )
                                
                                # 匹配 config_name（dataset_name）
                                # TaskManager 中的 dataset_name 对应文件夹中的 config_name
                                # 如果 TaskManager 中的 dataset_name 是 None，说明该任务不区分配置，可以匹配任何 config_name
                                # 如果 TaskManager 中的 dataset_name 不是 None，则必须与 config_name 匹配
                                # 特殊情况：如果本地 config_name 是 "all"，说明这是一个包含所有子任务的数据集，
                                # 可以匹配任何使用相同 dataset_path 的任务（优先匹配 dataset_name 为 None 的任务）
                                config_matches = False
                                if d_config == "all":
                                    # config_name 是 "all"，可以匹配任何使用相同 dataset_path 的任务
                                    # 优先匹配 dataset_name 为 None 的任务（通用任务）
                                    config_matches = True
                                elif dataset_name is None:
                                    # TaskManager 中没有 dataset_name，可以匹配任何 config_name（包括 None 和任何值）
                                    config_matches = True
                                elif dataset_name is not None and d_config is None:
                                    # TaskManager 有 dataset_name，但本地没有 config_name，只有当都为 None 时匹配
                                    config_matches = (dataset_name == d_config)
                                elif dataset_name is not None and d_config is not None:
                                    # 都有值，必须相同
                                    config_matches = (dataset_name == d_config)
                                
                                # 如果 path 和 config_name 都匹配，则找到
                                if path_matches and config_matches:
                                    # 如果 config_name 是 "all"，优先匹配 dataset_name 为 None 的任务（通用任务）
                                    if existing_index is None:
                                        existing_index = idx
                                        best_match_task_name = task_name
                                    elif d_config == "all" and dataset_name is None:
                                        # 对于 "all" 配置，优先匹配 dataset_name 为 None 的任务
                                        existing_index = idx
                                        best_match_task_name = task_name
                                    break
                            
                            if existing_index is not None:
                                # 如果已存在，更新 task_name 字段（从 TaskManager 获取），并补充其他信息
                                existing_dataset = datasets[existing_index]
                                # 更新数据集名称：使用 YAML 配置中的 task 字段
                                if task_name_from_config:
                                    existing_dataset["name"] = task_name_from_config
                                # 更新 task_name 字段为正确的任务名称（从 TaskManager 获取，用于评测）
                                # 注意：如果 config_name 是 "all"，可能需要匹配多个任务，这里优先匹配 dataset_name 为 None 的任务
                                if existing_dataset.get("task_name") is None:
                                    existing_dataset["task_name"] = best_match_task_name or task_name
                                # 如果 config_name 是 "all" 且当前任务的 dataset_name 为 None，优先使用它
                                elif existing_dataset.get("config_name") == "all" and dataset_name is None:
                                    existing_dataset["task_name"] = task_name
                                # 更新其他可能缺失的字段
                                if not existing_dataset.get("splits") or not existing_dataset.get("num_examples"):
                                    # 注意：dataset_path 可能使用斜杠（如 cais/mmlu），需要转换为下划线格式来匹配本地路径
                                    # 但 existing_dataset["path"] 已经是文件夹名称（下划线格式），应该使用它
                                    local_path = get_local_dataset_path(existing_dataset["path"], dataset_name)
                                    if local_path.exists():
                                        try:
                                            dataset = load_from_disk(str(local_path))
                                            splits = list(dataset.keys())
                                            num_examples = {split: len(dataset[split]) for split in splits}
                                            existing_dataset["splits"] = splits
                                            existing_dataset["num_examples"] = num_examples
                                            existing_dataset["is_local"] = True
                                            existing_dataset["local_path"] = str(local_path)
                                        except Exception:
                                            pass
                                # 更新标签和类别
                                tags = []
                                if "tag" in config:
                                    tag_value = config["tag"]
                                    if isinstance(tag_value, str):
                                        tags = [tag_value]
                                    elif isinstance(tag_value, list):
                                        tags = tag_value
                                existing_dataset["tags"] = tags
                                existing_dataset["category"] = infer_category(dataset_path, task_name, tags)
                            else:
                                # dataset_path 可能使用斜杠（如 cais/mmlu），需要转换为下划线格式来匹配本地路径
                                local_path = get_local_dataset_path(dataset_path.replace("/", "_"), dataset_name)
                                is_local = local_path.exists()
                                
                                # 如果数据集在本地，尝试加载获取 splits 和 num_examples
                                splits = None
                                num_examples = None
                                if is_local:
                                    try:
                                        dataset = load_from_disk(str(local_path))
                                        splits = list(dataset.keys())
                                        num_examples = {split: len(dataset[split]) for split in splits}
                                    except Exception:
                                        # 如果加载失败，尝试其他可能的路径格式
                                        # 例如：task_name 可能是 super_glue_boolq，但目录可能是 super_glue/boolq
                                        try:
                                            # 尝试将 task_name 中的下划线转换为路径
                                            task_name_parts = task_name.split("_")
                                            for i in range(len(task_name_parts) - 1, 0, -1):
                                                possible_path = "_".join(task_name_parts[:i])
                                                possible_config = "_".join(task_name_parts[i:])
                                                possible_local_path = get_local_dataset_path(possible_path, possible_config)
                                                if possible_local_path.exists():
                                                    dataset = load_from_disk(str(possible_local_path))
                                                    splits = list(dataset.keys())
                                                    num_examples = {split: len(dataset[split]) for split in splits}
                                                    local_path = possible_local_path
                                                    is_local = True
                                                    break
                                        except Exception:
                                            # 如果所有尝试都失败，保持为 None
                                            pass
                                
                                # 获取标签
                                tags = []
                                if "tag" in config:
                                    tag_value = config["tag"]
                                    if isinstance(tag_value, str):
                                        tags = [tag_value]
                                    elif isinstance(tag_value, list):
                                        tags = tag_value
                                
                                # 数据集名称：使用 YAML 配置中的 task 字段
                                # 如果没有 task 字段，则使用 task_name 作为后备
                                dataset_display_name = task_name_from_config if task_name_from_config else task_name
                                
                                # 创建数据集对象
                                dataset_info = {
                                    "name": dataset_display_name,  # 使用 YAML 配置中的 task 字段作为数据集名称
                                    "path": dataset_path,  # 保持原始路径（可能包含斜杠，用于匹配）
                                    "config_name": dataset_name,
                                    "task_name": task_name,  # 从 TaskManager 获取的任务名称（用于评测）
                                    "description": f"Task: {task_name}",
                                    "local_path": str(local_path) if is_local else None,
                                    "is_local": is_local,
                                    "splits": splits,
                                    "num_examples": num_examples,
                                    "category": infer_category(dataset_path, task_name, tags),
                                    "tags": tags
                                }
                                
                                # 合并已保存的元数据（如果存在）
                                metadata_key = f"{dataset_path}:{dataset_name}" if dataset_name else dataset_path
                                if metadata_key in saved_metadata:
                                    saved_meta = saved_metadata[metadata_key]
                                    # 合并元数据，已保存的优先级更高
                                    if "description" in saved_meta:
                                        dataset_info["description"] = saved_meta["description"]
                                    if "category" in saved_meta:
                                        dataset_info["category"] = saved_meta["category"]
                                    if "tags" in saved_meta:
                                        dataset_info["tags"] = saved_meta["tags"]
                                
                                datasets.append(dataset_info)
                    except Exception:
                        pass
            
            # 处理组（groups）- 组也可能对应数据集
            for group_name in task_manager.all_groups:
                task_info = task_manager.task_index.get(group_name, {})
                yaml_path = task_info.get("yaml_path", -1)
                
                if yaml_path != -1:
                    try:
                        from lm_eval import utils
                        config = utils.load_yaml_config(yaml_path, mode="simple")
                        dataset_path = config.get("dataset_path")
                        
                        # 对于组配置，task 字段是列表，但我们使用 group 名称作为数据集名称
                        # group 名称作为 task_name（用于评测），group 名称也作为数据集显示名称
                        task_name_from_config = group_name  # 组名作为数据集名称
                        
                        if dataset_path:
                            dataset_name = config.get("dataset_name")
                            
                            # 检查是否已在列表中（匹配 path 和 config_name）
                            existing_index = None
                            for idx, d in enumerate(datasets):
                                d_path = d.get("path")
                                d_config = d.get("config_name")
                                
                                # 匹配 path
                                path_matches = (
                                    d_path == dataset_path or
                                    d_path == dataset_path.replace("/", "_") or
                                    d_path.replace("_", "/") == dataset_path or
                                    d_path.replace("_", "/") == dataset_path.replace("/", "_")
                                )
                                
                                # 匹配 config_name（组配置通常没有 config_name，或者 config_name 是 "all"）
                                config_matches = (
                                    (dataset_name is None and d_config is None) or
                                    (dataset_name is not None and d_config == dataset_name) or
                                    d_config == "all"  # config_name 为 "all" 可以匹配组
                                )
                                
                                if path_matches and config_matches:
                                    existing_index = idx
                                    break
                            
                            if existing_index is not None:
                                # 如果已存在，更新为组信息
                                existing_dataset = datasets[existing_index]
                                # 更新数据集名称：使用组名
                                existing_dataset["name"] = task_name_from_config
                                # 更新 task_name 为组名（用于评测）
                                existing_dataset["task_name"] = group_name
                            else:
                                # 检查本地是否存在对应的数据集
                                local_path = get_local_dataset_path(dataset_path.replace("/", "_"), dataset_name)
                                is_local = local_path.exists()
                                
                                # 如果数据集在本地，尝试加载获取 splits 和 num_examples
                                splits = None
                                num_examples = None
                                if is_local:
                                    try:
                                        dataset = load_from_disk(str(local_path))
                                        splits = list(dataset.keys())
                                        num_examples = {split: len(dataset[split]) for split in splits}
                                    except Exception:
                                        pass
                                
                                # 获取标签
                                tags = []
                                if "tag" in config:
                                    tag_value = config["tag"]
                                    if isinstance(tag_value, str):
                                        tags = [tag_value]
                                    elif isinstance(tag_value, list):
                                        tags = tag_value
                                
                                # 创建数据集对象（组）
                                dataset_info = {
                                    "name": task_name_from_config,  # 使用组名作为数据集名称
                                    "path": dataset_path,
                                    "config_name": dataset_name,
                                    "task_name": group_name,  # 组名作为任务名称（用于评测）
                                    "description": f"Group: {group_name}",
                                    "local_path": str(local_path) if is_local else None,
                                    "is_local": is_local,
                                    "splits": splits,
                                    "num_examples": num_examples,
                                    "category": infer_category(dataset_path, group_name, tags),
                                    "tags": tags
                                }
                                
                                # 合并已保存的元数据（如果存在）
                                metadata_key = f"{dataset_path}:{dataset_name}" if dataset_name else dataset_path
                                if metadata_key in saved_metadata:
                                    saved_meta = saved_metadata[metadata_key]
                                    if "description" in saved_meta:
                                        dataset_info["description"] = saved_meta["description"]
                                    if "category" in saved_meta:
                                        dataset_info["category"] = saved_meta["category"]
                                    if "tags" in saved_meta:
                                        dataset_info["tags"] = saved_meta["tags"]
                                
                                datasets.append(dataset_info)
                    except Exception:
                        pass
        except Exception:
            pass
        
        # 确保所有数据集都有 name 字段（应该已经有了，但为了安全起见）
        # 如果 name 为 None，使用 task_name（从 TaskManager 获取的任务名称）作为后备
        # 如果 task_name 也没有，则使用 path（将斜杠转换为下划线）作为最后的后备
        for dataset in datasets:
            if dataset.get("name") is None:
                # 优先使用 task_name
                if dataset.get("task_name"):
                    dataset["name"] = dataset["task_name"]
                else:
                    # 如果 task_name 也没有，使用 path 作为后备
                    path_name = dataset.get("path", "").replace("/", "_")  # 将斜杠转换为下划线
                    if dataset.get("config_name"):
                        dataset["name"] = f"{path_name}_{dataset['config_name']}"
                    else:
                        dataset["name"] = path_name
        
        _datasets_cache = datasets
        return datasets


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    category: Optional[str] = Query(None, description="按类别过滤"),
    is_local: Optional[bool] = Query(None, description="是否只显示本地数据集"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    """获取数据集列表（支持分类过滤和分页）"""
    all_datasets = load_all_datasets()
    
    # 过滤
    filtered = all_datasets
    
    if category:
        filtered = [d for d in filtered if d.get("category") == category]
    
    if is_local is not None:
        filtered = [d for d in filtered if d.get("is_local") == is_local]
    
    if search:
        search_lower = search.lower()
        filtered = [
            d for d in filtered
            if search_lower in d.get("name", "").lower() 
            or search_lower in d.get("path", "").lower()
            or search_lower in d.get("description", "").lower()
        ]
    
    # 获取所有类别
    categories = sorted(set(d.get("category", "其他") for d in all_datasets))
    
    # 分页
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
    """刷新数据集缓存"""
    global _datasets_cache
    with _cache_lock:
        _datasets_cache = None
    # 强制重新加载一次，确保缓存被清除并重新构建
    try:
        load_all_datasets()
        return {"message": "缓存已刷新，数据集已重新加载"}
    except Exception as e:
        return {"message": f"缓存已清除，但重新加载时出错: {str(e)}"}


@router.get("/samples")
async def get_dataset_samples(dataset_name: str = Query(..., description="数据集名称"), split: str = Query("train"), limit: int = Query(2)):
    """获取数据集样本（带缓存）"""
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
    
    # 调试：打印所有数据集名称
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"查找数据集: {dataset_name}")
    logger.info(f"可用数据集名称: {[d.get('name') for d in all_datasets[:10]]}")
    
    # 首先尝试精确匹配 name
    for d in all_datasets:
        if d.get("name") == dataset_name:
            dataset_info = d
            logger.info(f"精确匹配成功: {dataset_name} -> {d.get('name')}")
            break
    
    # 如果找不到，尝试多种匹配方式
    if not dataset_info:
        # 将数据集名称中的斜杠转换为下划线，用于匹配目录名格式
        dataset_name_normalized = dataset_name.replace("/", "_")
        # 也尝试将下划线转换为斜杠
        dataset_name_with_slash = dataset_name.replace("_", "/")
        
        logger.info(f"尝试模糊匹配: {dataset_name} -> {dataset_name_normalized} 或 {dataset_name_with_slash}")
        
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
                logger.info(f"模糊匹配成功: {dataset_name} -> {name} (path: {path}, config: {config_name})")
                break
    
    if not dataset_info:
        # 提供更详细的错误信息，列出所有可用的数据集名称
        available_names = [d.get("name", "unknown") for d in all_datasets[:20]]  # 显示前20个
        # 也显示路径信息以便调试
        available_info = [
            f"{d.get('name', 'unknown')} (path: {d.get('path', 'unknown')}, config: {d.get('config_name', 'None')})"
            for d in all_datasets[:20]
        ]
        # 尝试查找包含相似路径的数据集
        similar_datasets = [
            d for d in all_datasets 
            if (dataset_name.replace("/", "_") in d.get("name", "") or 
                dataset_name.replace("_", "/") in d.get("name", "") or
                dataset_name in d.get("path", "") or
                dataset_name.replace("/", "_") in d.get("path", ""))
        ]
        similar_info = [
            f"{d.get('name', 'unknown')} (path: {d.get('path', 'unknown')})"
            for d in similar_datasets[:10]
        ]
        
        error_msg = f"数据集不存在: {dataset_name}"
        if similar_info:
            error_msg += f"。相似数据集: {'; '.join(similar_info)}"
        error_msg += f"。可用数据集（前10个）: {', '.join(available_names[:10])}"
        
        logger.error(f"数据集查找失败: {dataset_name}")
        logger.error(f"所有可用数据集: {available_info[:10]}")
        
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
    
    # 从本地加载数据集
    # load_from_disk 可以正确读取 Arrow 格式的数据集（data-00000-of-00001.arrow 等文件）
    try:
        # 确保路径是字符串格式
        dataset_path_str = str(local_path)
        
        # 使用 load_from_disk 加载数据集
        # 这会自动读取目录下的所有 splits（test/, train/, validation/ 等）
        # 以及对应的 Arrow 文件（data-00000-of-00001.arrow 等）
        dataset = load_from_disk(dataset_path_str)
        
        # 验证数据集是否成功加载
        if not dataset or not isinstance(dataset, dict):
            raise ValueError(f"数据集加载失败：返回的数据不是字典格式")
            
    except Exception as e:
        import traceback
        error_detail = f"加载本地数据集失败: {str(e)}。路径: {local_path}"
        # 在开发环境中，可以包含更详细的错误信息
        if os.environ.get("DEBUG", "").lower() == "true":
            error_detail += f"\n详细错误: {traceback.format_exc()}"
        raise HTTPException(
            status_code=500, 
            detail=error_detail
        )
    
    # 检查 split 是否存在
    if split not in dataset:
        available_splits = list(dataset.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Split '{split}' 不存在。可用 splits: {', '.join(available_splits)}"
        )
    
    # 获取样本
    # dataset[split] 返回的是 Dataset 对象，可以通过切片获取样本
    try:
        split_dataset = dataset[split]
        
        # 获取指定数量的样本
        # 使用 [:limit] 切片，这会返回 Dataset 对象的前 limit 条记录
        samples = split_dataset[:limit]
        
        # 转换为字典列表
        # 如果 samples 是字典（单个样本），转换为列表
        if isinstance(samples, dict):
            sample_list = [samples]
        else:
            # samples 是 Dataset 对象，转换为字典列表
            sample_list = [dict(sample) for sample in samples]
            
    except Exception as e:
        import traceback
        error_detail = f"读取样本失败: {str(e)}"
        if os.environ.get("DEBUG", "").lower() == "true":
            error_detail += f"\n详细错误: {traceback.format_exc()}"
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )
    
    # 更新缓存
    with _samples_cache_lock:
        _samples_cache[cache_key] = sample_list
    
    return sample_list


@router.get("/{dataset_name}", response_model=DatasetResponse)
async def get_dataset(dataset_name: str):
    """获取单个数据集详情"""
    # 如果 dataset_name 包含 '/'，说明可能是路由匹配错误，应该匹配到更具体的路由
    if '/' in dataset_name:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
    # 首先从缓存的数据集列表中查找（确保返回的 name 与列表中的一致）
    all_datasets = load_all_datasets()
    for dataset_info in all_datasets:
        if dataset_info["name"] == dataset_name:
            return DatasetResponse(**dataset_info)
    
    # 如果没找到，尝试从本地查找（兼容旧代码）
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            if dataset_dir.name == dataset_name.replace("/", "_"):
                try:
                    dataset = load_from_disk(str(dataset_dir))
                    splits = list(dataset.keys())
                    num_examples = {split: len(dataset[split]) for split in splits}
                    
                    return DatasetResponse(
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
                name=dataset_name,
                path=dataset_name,
                is_local=False,
                description=info.description,
                splits=list(info.splits.keys()) if info.splits else None
            )
    except Exception:
        pass
    
    raise HTTPException(status_code=404, detail="数据集不存在")


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
        
        # 尝试从 TaskManager 获取正确的任务名称和 task 字段
        task_name = None
        task_from_config = None
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
                            # 获取 YAML 配置中的 task 字段作为数据集名称
                            task_from_config = config.get("task")
                            break
                    except Exception:
                        pass
        except Exception:
            pass
        
        # 如果没有找到匹配的任务名称，使用 path_config_name 格式作为后备
        if task_name is None:
            task_name = request.dataset_path + (f"_{request.dataset_name}" if request.dataset_name else "")
        
        # 数据集名称：优先使用 YAML 配置中的 task 字段，如果没有则使用 task_name
        dataset_display_name = task_from_config if task_from_config else task_name
        
        # 构建数据集响应
        dataset_response = DatasetResponse(
            name=dataset_display_name,  # 使用 YAML 配置中的 task 字段作为数据集名称
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


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """删除本地数据集"""
    global _datasets_cache
    
    # 先查找数据集信息，以便删除元数据
    all_datasets = load_all_datasets()
    dataset_info = None
    for ds in all_datasets:
        if ds.get("name") == dataset_name:
            dataset_info = ds
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

