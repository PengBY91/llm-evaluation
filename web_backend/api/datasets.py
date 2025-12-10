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

router = APIRouter()

# 数据集存储目录
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

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
    name: str
    path: str
    config_name: Optional[str] = None
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
        if _datasets_cache is not None:
            return _datasets_cache
        
        datasets = []
        
        # 扫描本地 data 目录
        if DATA_DIR.exists():
            for dataset_dir in DATA_DIR.iterdir():
                if dataset_dir.is_dir():
                    config_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
                    
                    if config_dirs:
                        for config_dir in config_dirs:
                            dataset_path = dataset_dir.name.replace("_", "/")
                            config_name = config_dir.name
                            
                            try:
                                dataset = load_from_disk(str(config_dir))
                                splits = list(dataset.keys())
                                num_examples = {split: len(dataset[split]) for split in splits}
                                
                                datasets.append({
                                    "name": f"{dataset_path}_{config_name}",
                                    "path": dataset_path,
                                    "config_name": config_name,
                                    "local_path": str(config_dir),
                                    "is_local": True,
                                    "splits": splits,
                                    "num_examples": num_examples,
                                    "category": infer_category(dataset_path),
                                    "tags": []
                                })
                            except Exception:
                                pass
                    else:
                        dataset_path = dataset_dir.name.replace("_", "/")
                        
                        try:
                            dataset = load_from_disk(str(dataset_dir))
                            splits = list(dataset.keys())
                            num_examples = {split: len(dataset[split]) for split in splits}
                            
                            datasets.append({
                                "name": dataset_path,
                                "path": dataset_path,
                                "config_name": None,
                                "local_path": str(dataset_dir),
                                "is_local": True,
                                "splits": splits,
                                "num_examples": num_examples,
                                "category": infer_category(dataset_path),
                                "tags": []
                            })
                        except Exception:
                            pass
        
        # 从 TaskManager 获取所有可用任务（数据集）
        try:
            from lm_eval.tasks import TaskManager
            task_manager = TaskManager()
            
            for task_name in task_manager.all_subtasks:
                task_info = task_manager.task_index.get(task_name, {})
                yaml_path = task_info.get("yaml_path", -1)
                
                if yaml_path != -1:
                    try:
                        from lm_eval import utils
                        config = utils.load_yaml_config(yaml_path, mode="simple")
                        dataset_path = config.get("dataset_path")
                        
                        if dataset_path:
                            dataset_name = config.get("dataset_name")
                            
                            # 检查是否已在列表中
                            existing = any(
                                d["path"] == dataset_path and d.get("config_name") == dataset_name
                                for d in datasets
                            )
                            
                            if not existing:
                                local_path = get_local_dataset_path(dataset_path, dataset_name)
                                is_local = local_path.exists()
                                
                                # 获取标签
                                tags = []
                                if "tag" in config:
                                    tag_value = config["tag"]
                                    if isinstance(tag_value, str):
                                        tags = [tag_value]
                                    elif isinstance(tag_value, list):
                                        tags = tag_value
                                
                                datasets.append({
                                    "name": task_name,
                                    "path": dataset_path,
                                    "config_name": dataset_name,
                                    "description": f"Task: {task_name}",
                                    "local_path": str(local_path) if is_local else None,
                                    "is_local": is_local,
                                    "splits": None,
                                    "num_examples": None,
                                    "category": infer_category(dataset_path, task_name, tags),
                                    "tags": tags
                                })
                    except Exception:
                        pass
        except Exception:
            pass
        
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
    return {"message": "缓存已刷新"}


@router.get("/{dataset_name}", response_model=DatasetResponse)
async def get_dataset(dataset_name: str):
    """获取单个数据集详情"""
    # 首先尝试从本地查找
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
        
        # 清除缓存
        with _cache_lock:
            _datasets_cache = None
        
        return DatasetResponse(
            name=request.dataset_path + (f"_{request.dataset_name}" if request.dataset_name else ""),
            path=request.dataset_path,
            config_name=request.dataset_name,
            description=request.description,
            local_path=local_path,
            is_local=request.save_local,
            splits=splits,
            num_examples=num_examples,
            category=infer_category(request.dataset_path),
            tags=[]
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"添加数据集失败: {str(e)}")


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """删除本地数据集"""
    global _datasets_cache
    
    dataset_path = DATA_DIR / dataset_name.replace("/", "_")
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="本地数据集不存在")
    
    import shutil
    try:
        shutil.rmtree(dataset_path)
        
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


@router.get("/{dataset_name}/samples")
async def get_dataset_samples(dataset_name: str, split: str = "train", limit: int = 10):
    """获取数据集样本（带缓存）"""
    cache_key = f"{dataset_name}:{split}"
    
    # 检查缓存
    with _samples_cache_lock:
        if cache_key in _samples_cache and len(_samples_cache[cache_key]) >= limit:
            return _samples_cache[cache_key][:limit]
    
    # 加载数据集
    dataset_path = DATA_DIR / dataset_name.replace("/", "_")
    
    if dataset_path.exists():
        # 从本地加载
        dataset = load_from_disk(str(dataset_path))
    else:
        # 从 HuggingFace 加载
        try:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
        except Exception:
            raise HTTPException(status_code=404, detail="数据集不存在")
    
    if split not in dataset:
        raise HTTPException(status_code=404, detail=f"Split '{split}' 不存在")
    
    samples = dataset[split][:limit]
    
    # 转换为字典列表
    sample_list = [dict(sample) for sample in samples]
    
    # 更新缓存
    with _samples_cache_lock:
        _samples_cache[cache_key] = sample_list
    
    return sample_list

