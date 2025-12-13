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

# 数据集缓存
_datasets_cache: Optional[List[Dict[str, Any]]] = None
_cache_lock = threading.Lock()

# 数据集样本缓存
_samples_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_samples_cache_lock = threading.Lock()


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
    output_type: Optional[str] = None  # 任务输出类型 (e.g. multiple_choice, generate_until)


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
    tags: Optional[List[str]] = None  # 标签列表
    output_type: Optional[str] = None  # 任务输出类型


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
    """重建数据集索引（扫描磁盘并保存到文件）"""
    global _datasets_cache
    
    # 加载已保存的数据集元数据
    saved_metadata = load_all_datasets_metadata()
    
    datasets = []
    
    # 1. 扫描本地 data 目录，建立数据映射表
    # local_data_map: {(path, config) -> local_path}
    local_data_map = {}
    
    if DATA_DIR.exists():
        for root, dirs, files in os.walk(DATA_DIR):
            root_path = Path(root)
            
            # 跳过 DATA_DIR 本身和特殊目录
            if root_path == DATA_DIR or any(part.startswith('.') for part in root_path.parts) or \
               'datasets_metadata' in root_path.parts or 'tasks' in root_path.parts:
                continue
            
            # 检查是否是有效数据集
            has_dataset_dict = 'dataset_dict.json' in files
            has_dataset_info = 'dataset_info.json' in files
            has_arrow = any(f.endswith('.arrow') for f in files)
            
            is_potential_dataset = has_dataset_dict or has_dataset_info or has_arrow

            if is_potential_dataset:
                try:
                    rel_path = root_path.relative_to(DATA_DIR)
                    parts = rel_path.parts
                    
                    dataset_path = None
                    config_name = None
                    
                    # 尝试解析路径
                    if len(parts) == 1:
                        dataset_path = parts[0]
                    elif len(parts) >= 2:
                        dataset_path = "/".join(parts[:-1])
                        config_name = parts[-1]
                    
                    # 存储映射
                    if dataset_path:
                         # 原始路径
                         local_data_map[(dataset_path, config_name)] = str(root_path)
                         # 简化路径（处理 cais/mmlu vs mmlu）
                         if "/" in dataset_path:
                             simple_path = dataset_path.split("/")[-1]
                             local_data_map[(simple_path, config_name)] = str(root_path)
                             
                except Exception:
                    pass

            # 如果当前目录被识别为数据集（无论是否成功解析路径），则不再遍历其子目录
            # 这可以防止将数据集的子文件夹（如 splits, checkpoints）误识别为独立数据集
            if is_potential_dataset:
                dirs[:] = []

    # 2. 从 TaskManager 获取所有任务，并匹配本地数据
    try:
        from lm_eval.tasks import TaskManager
        from lm_eval import utils
        task_manager = TaskManager()
        
        # 记录已处理的本地路径，用于最后找出未匹配的数据集
        processed_local_paths = set()
        
        # 遍历所有子任务
        for task_name in task_manager.all_subtasks:
            task_info = task_manager.task_index.get(task_name, {})
            yaml_path = task_info.get("yaml_path", -1)
            
            if yaml_path != -1:
                try:
                    config = utils.load_yaml_config(yaml_path, mode="simple")
                    ds_path = config.get("dataset_path")
                    ds_name = config.get("dataset_name")
                    output_type = config.get("output_type")
                    
                    # 查找匹配的本地数据
                    matched_local_path = None
                    
                    # 尝试匹配
                    # A. 精确匹配
                    if (ds_path, ds_name) in local_data_map:
                        matched_local_path = local_data_map[(ds_path, ds_name)]
                    # B. 简化路径匹配 (cais/mmlu -> mmlu)
                    elif ds_path and "/" in ds_path and (ds_path.split("/")[-1], ds_name) in local_data_map:
                         matched_local_path = local_data_map[(ds_path.split("/")[-1], ds_name)]
                    # C. None config 匹配 (dataset_name is None -> config is None or 'default'?)
                    elif ds_name is None:
                         if (ds_path, None) in local_data_map:
                             matched_local_path = local_data_map[(ds_path, None)]
                         elif ds_path and "/" in ds_path and (ds_path.split("/")[-1], None) in local_data_map:
                             matched_local_path = local_data_map[(ds_path.split("/")[-1], None)]
                    
                    if matched_local_path:
                        processed_local_paths.add(matched_local_path)
                        
                        # 加载数据集统计信息
                        splits = None
                        num_examples = None
                        try:
                            ds_obj = load_from_disk(matched_local_path)
                            splits = list(ds_obj.keys())
                            num_examples = {split: len(ds_obj[split]) for split in splits}
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

                        # 格式化显示名称
                        display_name = task_name
                        if ds_name and ds_name not in ["default", "main"]:
                             # 如果任务名已经包含了配置名（如 mmlu_abstract_algebra），尝试让显示更友好
                             if task_name.endswith(f"_{ds_name}"):
                                 base = task_name[:-len(ds_name)-1]
                                 display_name = f"{base} ({ds_name})"
                             elif ds_name not in task_name:
                                 display_name = f"{task_name} ({ds_name})"

                        dataset_info = {
                            "id": task_name,  # 使用 task_name 作为 ID，确保唯一性且与评测一致
                            "name": display_name, # 显示名称
                            "path": ds_path or task_name,
                            "config_name": ds_name,
                            "task_name": task_name,
                            "description": f"Task: {task_name}",
                            "local_path": matched_local_path,
                            "is_local": True,
                            "splits": splits,
                            "num_examples": num_examples,
                            "category": infer_category(ds_path or task_name, task_name, tags),
                            "tags": tags,
                            "output_type": output_type
                        }
                        
                        # 合并已保存的元数据
                        metadata_key = task_name # 使用 task_name 作为 key 更可靠
                        if metadata_key in saved_metadata:
                             saved_meta = saved_metadata[metadata_key]
                             if "description" in saved_meta: dataset_info["description"] = saved_meta["description"]
                             if "category" in saved_meta: dataset_info["category"] = saved_meta["category"]
                             if "tags" in saved_meta: dataset_info["tags"] = saved_meta["tags"]
                        
                        # 兼容旧 key
                        old_key = f"{ds_path}:{ds_name}" if ds_name else ds_path
                        if old_key in saved_metadata:
                             saved_meta = saved_metadata[old_key]
                             if "description" in saved_meta: dataset_info["description"] = saved_meta["description"]
                             if "category" in saved_meta: dataset_info["category"] = saved_meta["category"]
                             if "tags" in saved_meta: dataset_info["tags"] = saved_meta["tags"]

                        datasets.append(dataset_info)
                except Exception:
                    pass

        # 3. 处理任务组 (Groups)
        # 如果一个组对应的文件夹存在，我们也应该允许运行整个组
        for group_name in task_manager.all_groups:
             # 检查是否有对应的本地数据 (group_name, None)
             matched_local_path = None
             if (group_name, None) in local_data_map:
                 matched_local_path = local_data_map[(group_name, None)]
             
             if matched_local_path and group_name not in [d["id"] for d in datasets]:
                 processed_local_paths.add(matched_local_path)
                 # ... 创建 Group 的 dataset_info
                 try:
                    ds_obj = load_from_disk(matched_local_path)
                    splits = list(ds_obj.keys())
                    num_examples = {split: len(ds_obj[split]) for split in splits}
                 except Exception:
                    splits, num_examples = None, None

                 dataset_info = {
                    "id": group_name,
                    "name": group_name,
                    "path": group_name,
                    "config_name": None,
                    "task_name": group_name,
                    "description": f"Task Group: {group_name}",
                    "local_path": matched_local_path,
                    "is_local": True,
                    "splits": splits,
                    "num_examples": num_examples,
                    "category": infer_category(group_name),
                    "tags": ["group"],
                    "output_type": None # Group 没有单一 output_type
                }
                 datasets.append(dataset_info)

        # 4. 处理未匹配的本地数据集 (Fallback)
        # 这些可能是自定义数据集，或者 TaskManager 中没有定义的
        for (d_path, d_config), local_path in local_data_map.items():
            if local_path not in processed_local_paths:
                # 这是一个未被任何 Task 认领的数据集
                # 我们仍然显示它，但在 task_name 上可能为空，或者尽力推断
                
                dataset_name = d_path.replace("/", "_")
                if d_config:
                    dataset_name += f"_{d_config}"
                
                splits, num_examples = None, None
                try:
                    ds_obj = load_from_disk(local_path)
                    splits = list(ds_obj.keys())
                    num_examples = {split: len(ds_obj[split]) for split in splits}
                except Exception:
                    pass
                
                dataset_info = {
                    "id": generate_dataset_id(d_path, d_config),
                    "name": dataset_name,
                    "path": d_path,
                    "config_name": d_config,
                    "task_name": None, # 无法确定 Task
                    "description": "Unknown Task / Custom Dataset",
                    "local_path": local_path,
                    "is_local": True,
                    "splits": splits,
                    "num_examples": num_examples,
                    "category": infer_category(d_path),
                    "tags": ["custom"],
                    "output_type": None
                }
                datasets.append(dataset_info)

    except Exception as e:
        print(f"Rebuild index error: {e}")
        # 如果 TaskManager 失败，回退到原来的简单扫描逻辑
        # (这里为了简洁省略回退代码，实际生产中应保留)
        pass
    
    # 排序：有 task_name 的排前面，然后按名称排
    datasets.sort(key=lambda x: (x["task_name"] is None, x["name"]))
    
    # 保存到索引文件
    try:
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(datasets, f, ensure_ascii=False, indent=2)
        print(f"数据集索引已保存到: {INDEX_FILE}")
    except Exception as e:
        print(f"保存数据集索引失败: {e}")
    
    _datasets_cache = datasets
    return datasets


def load_all_datasets() -> List[Dict[str, Any]]:
    """加载所有数据集（带缓存）"""
    global _datasets_cache
    
    with _cache_lock:
        # 如果缓存存在，直接返回（避免重复扫描）
        if _datasets_cache is not None:
            return _datasets_cache
        
        datasets = None
        
        # 尝试从索引文件加载
        if INDEX_FILE.exists():
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    datasets = json.load(f)
            except Exception as e:
                print(f"读取数据集索引文件失败: {e}，将重新扫描")
        
        # 如果没有索引文件或读取失败，重建索引
        if datasets is None:
            return rebuild_dataset_index()
            
        # 检查并修复缺失的 ID (兼容旧索引文件)
        needs_save = False
        for dataset in datasets:
            if dataset.get("id") is None:
                dataset["id"] = generate_dataset_id(dataset.get("path", ""), dataset.get("config_name"))
                needs_save = True
        
        # 如果有更新，保存回文件
        if needs_save:
            try:
                with open(INDEX_FILE, "w", encoding="utf-8") as f:
                    json.dump(datasets, f, ensure_ascii=False, indent=2)
                print(f"数据集索引已更新并保存到: {INDEX_FILE}")
            except Exception as e:
                print(f"保存更新后的数据集索引失败: {e}")
        
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
    """刷新数据集缓存（触发重新扫描并更新索引）"""
    global _datasets_cache
    with _cache_lock:
        _datasets_cache = None
    
    # 强制重新扫描并更新索引文件
    try:
        rebuild_dataset_index()
        return {"message": "缓存已刷新，数据集索引已更新"}
    except Exception as e:
        return {"message": f"缓存刷新失败: {str(e)}"}


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


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """获取单个数据集详情（支持 ID 或 Name）"""
    all_datasets = load_all_datasets()
    
    # 1. 优先匹配 ID
    for dataset_info in all_datasets:
        if dataset_info.get("id") == dataset_id:
            return DatasetResponse(**dataset_info)
    
    # 2. 兼容旧逻辑：匹配 Name
    # 如果 dataset_id 包含 '/'，说明可能是路由匹配错误，应该匹配到更具体的路由
    if '/' in dataset_id:
        raise HTTPException(status_code=404, detail="数据集不存在")
    
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

