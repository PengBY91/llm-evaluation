"""
模型管理 API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
from pathlib import Path

router = APIRouter()

# 模型配置存储目录
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 内存中的模型配置（实际应用中应使用数据库）
models_db: Dict[str, Dict[str, Any]] = {}


class ModelCreateRequest(BaseModel):
    """创建模型请求"""
    model_config = {"protected_namespaces": ()}
    
    name: str
    model_type: str  # 模型类型：openai-chat-completions, openai-completions, hf, vllm 等
    description: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    port: Optional[int] = None
    max_concurrent: Optional[int] = None  # 最大并发数
    max_tokens: Optional[int] = None  # 最长 token
    model_name: Optional[str] = None  # 模型名称（如 gpt-3.5-turbo）
    other_config: Optional[Dict[str, Any]] = None  # 其他配置项


class ModelUpdateRequest(BaseModel):
    """更新模型请求"""
    model_config = {"protected_namespaces": ()}
    
    name: Optional[str] = None
    model_type: Optional[str] = None
    description: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    port: Optional[int] = None
    max_concurrent: Optional[int] = None
    max_tokens: Optional[int] = None
    model_name: Optional[str] = None
    other_config: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    """模型响应"""
    id: str
    name: str
    model_type: str
    description: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None  # 返回时隐藏实际值
    port: Optional[int] = None
    max_concurrent: Optional[int] = None
    max_tokens: Optional[int] = None
    model_name: Optional[str] = None
    other_config: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


def save_model_to_file(model_id: str, model_data: Dict[str, Any]):
    """保存模型配置到文件"""
    model_file = MODELS_DIR / f"{model_id}.json"
    # 保存完整数据（包括 api_key）
    with open(model_file, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)


def load_model_from_file(model_id: str) -> Optional[Dict[str, Any]]:
    """从文件加载模型配置"""
    model_file = MODELS_DIR / f"{model_id}.json"
    if model_file.exists():
        with open(model_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_model_args(model: Dict[str, Any]) -> Dict[str, Any]:
    """根据模型配置生成 model_args"""
    model_args = {}
    
    if model.get("model_name"):
        model_args["model"] = model["model_name"]
    
    if model.get("base_url"):
        model_args["base_url"] = model["base_url"]
    
    if model.get("api_key"):
        model_args["api_key"] = model["api_key"]
    
    if model.get("max_concurrent"):
        model_args["num_concurrent"] = model["max_concurrent"]
    
    if model.get("port"):
        model_args["port"] = model["port"]
    
    if model.get("other_config"):
        model_args.update(model["other_config"])
    
    return model_args


@router.post("/", response_model=ModelResponse)
async def create_model(request: ModelCreateRequest):
    """创建新的模型配置"""
    model_id = str(uuid.uuid4())
    
    model_data = {
        "id": model_id,
        "name": request.name,
        "model_type": request.model_type,
        "description": request.description,
        "base_url": request.base_url,
        "api_key": request.api_key,
        "port": request.port,
        "max_concurrent": request.max_concurrent,
        "max_tokens": request.max_tokens,
        "model_name": request.model_name,
        "other_config": request.other_config or {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    
    models_db[model_id] = model_data
    save_model_to_file(model_id, model_data)
    
    # 返回时隐藏 api_key
    response_data = model_data.copy()
    if response_data.get("api_key"):
        response_data["api_key"] = "***"
    
    return ModelResponse(**response_data)


@router.get("/", response_model=List[ModelResponse])
async def list_models():
    """获取所有模型列表"""
    # 加载文件中的模型
    for model_file in MODELS_DIR.glob("*.json"):
        model_id = model_file.stem
        if model_id not in models_db:
            model_data = load_model_from_file(model_id)
            if model_data:
                models_db[model_id] = model_data
    
    # 返回时隐藏所有 api_key
    result = []
    for model in models_db.values():
        model_copy = model.copy()
        if model_copy.get("api_key"):
            model_copy["api_key"] = "***"
        result.append(ModelResponse(**model_copy))
    
    return result


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str):
    """获取单个模型详情"""
    if model_id not in models_db:
        model_data = load_model_from_file(model_id)
        if model_data:
            models_db[model_id] = model_data
        else:
            raise HTTPException(status_code=404, detail="模型不存在")
    
    model = models_db[model_id]
    
    # 返回时隐藏 api_key
    model_copy = model.copy()
    if model_copy.get("api_key"):
        model_copy["api_key"] = "***"
    
    return ModelResponse(**model_copy)


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(model_id: str, request: ModelUpdateRequest):
    """更新模型配置"""
    if model_id not in models_db:
        model_data = load_model_from_file(model_id)
        if model_data:
            models_db[model_id] = model_data
        else:
            raise HTTPException(status_code=404, detail="模型不存在")
    
    model_data = models_db[model_id]
    
    # 更新字段（只更新提供的字段）
    update_data = request.dict(exclude_unset=True)
    
    # 如果更新 api_key，需要特殊处理
    if "api_key" in update_data:
        # 如果传入的是 "***" 或空字符串，表示不更新，保持原有值
        if update_data["api_key"] == "***" or update_data["api_key"] == "":
            del update_data["api_key"]
        # 否则更新为新值
    
    for key, value in update_data.items():
        if value is not None:
            model_data[key] = value
    
    model_data["updated_at"] = datetime.now().isoformat()
    
    models_db[model_id] = model_data
    save_model_to_file(model_id, model_data)
    
    # 返回时隐藏 api_key
    response_data = model_data.copy()
    if response_data.get("api_key"):
        response_data["api_key"] = "***"
    
    return ModelResponse(**response_data)


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """删除模型配置"""
    if model_id not in models_db:
        model_data = load_model_from_file(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="模型不存在")
    
    # 删除文件
    model_file = MODELS_DIR / f"{model_id}.json"
    if model_file.exists():
        model_file.unlink()
    
    # 删除内存中的数据
    if model_id in models_db:
        del models_db[model_id]
    
    return {"message": "模型已删除"}


@router.get("/types/list")
async def get_model_types():
    """获取支持的模型类型列表"""
    return {
        "model_types": [
            {
                "value": "openai-chat-completions",
                "label": "OpenAI Chat Completions",
                "description": "OpenAI 兼容的聊天完成 API",
                "requires": ["base_url", "model_name"],
                "optional": ["api_key", "max_concurrent", "max_tokens"]
            },
            {
                "value": "openai-completions",
                "label": "OpenAI Completions",
                "description": "OpenAI 兼容的完成 API",
                "requires": ["base_url", "model_name"],
                "optional": ["api_key", "max_concurrent", "max_tokens"]
            },
            {
                "value": "hf",
                "label": "HuggingFace",
                "description": "HuggingFace Transformers 模型",
                "requires": ["model_name"],
                "optional": ["port", "max_tokens"]
            },
            {
                "value": "vllm",
                "label": "vLLM",
                "description": "vLLM 推理服务",
                "requires": ["base_url"],
                "optional": ["port", "max_concurrent", "max_tokens"]
            },
            {
                "value": "local-completions",
                "label": "Local Completions",
                "description": "本地完成服务",
                "requires": ["base_url"],
                "optional": ["port", "max_concurrent"]
            }
        ]
    }


@router.get("/{model_id}/model-args")
async def get_model_args_for_eval(model_id: str):
    """获取用于评测的 model_args（包含完整的 api_key）"""
    if model_id not in models_db:
        model_data = load_model_from_file(model_id)
        if model_data:
            models_db[model_id] = model_data
        else:
            raise HTTPException(status_code=404, detail="模型不存在")
    
    model = models_db[model_id]
    model_args = get_model_args(model)
    
    return {
        "model_type": model["model_type"],
        "model_args": model_args
    }

