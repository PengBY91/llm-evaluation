"""
评测任务配置管理 API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
from pathlib import Path

router = APIRouter()

# 配置存储目录
CONFIGS_DIR = Path(__file__).parent.parent.parent.parent / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

# 内存中的配置（实际应用中应使用数据库）
configs_db: Dict[str, Dict[str, Any]] = {}


class ConfigCreateRequest(BaseModel):
    """创建配置请求"""
    model_config = {"protected_namespaces": ()}
    
    name: str
    description: Optional[str] = None
    model: str
    model_args: Dict[str, Any]
    tasks: List[str]
    num_fewshot: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    limit: Optional[int] = None
    log_samples: bool = True
    apply_chat_template: Optional[bool] = False
    gen_kwargs: Optional[Dict[str, Any]] = None


class ConfigUpdateRequest(BaseModel):
    """更新配置请求"""
    model_config = {"protected_namespaces": ()}
    
    name: Optional[str] = None
    description: Optional[str] = None
    model: Optional[str] = None
    model_args: Optional[Dict[str, Any]] = None
    tasks: Optional[List[str]] = None
    num_fewshot: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    limit: Optional[int] = None
    log_samples: Optional[bool] = None
    apply_chat_template: Optional[bool] = None
    gen_kwargs: Optional[Dict[str, Any]] = None


class ConfigResponse(BaseModel):
    """配置响应"""
    id: str
    name: str
    description: Optional[str] = None
    model: str
    model_args: Dict[str, Any]
    tasks: List[str]
    num_fewshot: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    limit: Optional[int] = None
    log_samples: bool = True
    apply_chat_template: Optional[bool] = False
    gen_kwargs: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


def save_config_to_file(config_id: str, config_data: Dict[str, Any]):
    """保存配置到文件"""
    config_file = CONFIGS_DIR / f"{config_id}.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)


def load_config_from_file(config_id: str) -> Optional[Dict[str, Any]]:
    """从文件加载配置"""
    config_file = CONFIGS_DIR / f"{config_id}.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@router.post("/", response_model=ConfigResponse)
async def create_config(request: ConfigCreateRequest):
    """创建新的评测配置"""
    config_id = str(uuid.uuid4())
    
    config_data = {
        "id": config_id,
        "name": request.name,
        "description": request.description,
        "model": request.model,
        "model_args": request.model_args,
        "tasks": request.tasks,
        "num_fewshot": request.num_fewshot,
        "batch_size": request.batch_size,
        "device": request.device,
        "limit": request.limit,
        "log_samples": request.log_samples,
        "apply_chat_template": request.apply_chat_template,
        "gen_kwargs": request.gen_kwargs,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    
    configs_db[config_id] = config_data
    save_config_to_file(config_id, config_data)
    
    return ConfigResponse(**config_data)


@router.get("/", response_model=List[ConfigResponse])
async def list_configs():
    """获取所有配置列表"""
    # 加载文件中的配置
    for config_file in CONFIGS_DIR.glob("*.json"):
        config_id = config_file.stem
        if config_id not in configs_db:
            config_data = load_config_from_file(config_id)
            if config_data:
                configs_db[config_id] = config_data
    
    return [ConfigResponse(**config) for config in configs_db.values()]


@router.get("/{config_id}", response_model=ConfigResponse)
async def get_config(config_id: str):
    """获取单个配置详情"""
    if config_id not in configs_db:
        config_data = load_config_from_file(config_id)
        if config_data:
            configs_db[config_id] = config_data
        else:
            raise HTTPException(status_code=404, detail="配置不存在")
    
    return ConfigResponse(**configs_db[config_id])


@router.put("/{config_id}", response_model=ConfigResponse)
async def update_config(config_id: str, request: ConfigUpdateRequest):
    """更新配置"""
    if config_id not in configs_db:
        config_data = load_config_from_file(config_id)
        if config_data:
            configs_db[config_id] = config_data
        else:
            raise HTTPException(status_code=404, detail="配置不存在")
    
    config_data = configs_db[config_id]
    
    # 更新字段
    update_data = request.dict(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            config_data[key] = value
    
    config_data["updated_at"] = datetime.now().isoformat()
    
    configs_db[config_id] = config_data
    save_config_to_file(config_id, config_data)
    
    return ConfigResponse(**config_data)


@router.delete("/{config_id}")
async def delete_config(config_id: str):
    """删除配置"""
    if config_id not in configs_db:
        config_data = load_config_from_file(config_id)
        if not config_data:
            raise HTTPException(status_code=404, detail="配置不存在")
    
    # 删除文件
    config_file = CONFIGS_DIR / f"{config_id}.json"
    if config_file.exists():
        config_file.unlink()
    
    # 删除内存中的数据
    if config_id in configs_db:
        del configs_db[config_id]
    
    return {"message": "配置已删除"}

