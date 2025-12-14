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

# 内存中的模型配置（使用文件持久化）
models_db: Dict[str, Dict[str, Any]] = {}


def load_model_from_file(model_id: str) -> Optional[Dict[str, Any]]:
    """从文件加载模型配置"""
    model_file = MODELS_DIR / f"{model_id}.json"
    if model_file.exists():
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取模型文件失败 {model_id}: {e}")
            return None
    return None


def load_all_models_from_files():
    """从文件加载所有模型"""
    models = {}
    for model_file in MODELS_DIR.glob("*.json"):
        model_id = model_file.stem
        try:
            model_data = load_model_from_file(model_id)
            if model_data:
                models[model_id] = model_data
        except Exception as e:
            print(f"加载模型文件失败 {model_id}: {e}")
    return models


# 启动时加载所有模型
models_db = load_all_models_from_files()


class ModelCreateRequest(BaseModel):
    """创建模型请求"""
    model_config = {"protected_namespaces": ()}
    
    name: str
    model_type: str  # 模型类型：openai-chat-completions（支持 Ollama、vLLM 等）, openai-completions, hf
    description: Optional[str] = None
    base_url: Optional[str] = None  # 完整的 API URL，包含协议、主机、端口和路径
    api_key: Optional[str] = None
    port: Optional[int] = None  # 保留用于兼容旧数据，新数据应直接包含在 base_url 中
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
    model_config = {"protected_namespaces": ()}
    
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


def get_model_args(model: Dict[str, Any]) -> Dict[str, Any]:
    """根据模型配置生成 model_args"""
    model_args = {}
    
    if model.get("model_name"):
        model_args["model"] = model["model_name"]
    
    # base_url 应该包含完整的 URL（包括端口和路径）
    # 如果旧数据有单独的 port，需要合并
    base_url = model.get("base_url", "")
    port = model.get("port")
    model_type = model.get("model_type", "")
    
    if base_url:
        # 如果有单独的 port 且 base_url 中没有端口，则合并
        if port and ":" not in base_url.split("//")[1].split("/")[0]:
            from urllib.parse import urlparse, urlunparse
            try:
                parsed = urlparse(base_url)
                # 构建包含端口的新 URL
                netloc = f"{parsed.hostname}:{port}" if parsed.hostname else ""
                base_url = urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
            except Exception:
                # 如果解析失败，简单拼接
                if "://" in base_url:
                    parts = base_url.split("://", 1)
                    base_url = f"{parts[0]}://{parts[1].split('/')[0]}:{port}{'/' + '/'.join(parts[1].split('/')[1:]) if '/' in parts[1] else ''}"
                else:
                    base_url = f"{base_url}:{port}"
        
        # DeepSeek API 特殊处理：如果包含 api.deepseek.com，不要自动添加 /v1/chat/completions 后缀
        # 因为 DeepSeek 的 base_url 通常是 https://api.deepseek.com
        # 如果自动添加后缀，可能会变成 https://api.deepseek.com/v1/chat/completions/v1/chat/completions
        # lm-eval 库会自动处理 endpoint，我们只需要提供 base URL
        
        # 修正：对于 DeepSeek API，base_url 应该是 https://api.deepseek.com
        # lm-eval 库内部会自动调用 /chat/completions 端点
        # 但如果我们在这里添加了 /v1/chat/completions，lm-eval 可能会调用 https://api.deepseek.com/v1/chat/completions/chat/completions
        
        # 实际上，lm-eval 的 openai 接口需要的是 base_url，它会自动添加 /chat/completions (如果需要)
        # 但是，这取决于 lm-eval 版本和具体的接口实现。
        # 观察用户报错：Invalid URL (POST /v1) - 这表明请求发往了 /v1 而不是完整的 URL
        # 这通常意味着 base_url 解析有问题，或者 lm-eval 期望 base_url 不包含路径
        
        # 让我们检查一下 urlparse 的行为
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        
        # 如果 base_url 已经是完整的 API 端点（例如 https://api.deepseek.com/chat/completions），
        # 某些客户端可能会再次追加路径。
        # 但通常 base_url 应该是 https://api.deepseek.com/v1 或 https://api.deepseek.com
        
        # 针对 DeepSeek 的官方文档，base_url 是 https://api.deepseek.com
        # 如果用户填写的是 https://api.deepseek.com，我们可能不需要自动添加 /v1/chat/completions
        # 或者我们需要添加 /v1，变成 https://api.deepseek.com/v1
        
        # 如果 base_url 不包含路径，根据模型类型自动添加默认路径
        if not parsed.path or parsed.path == "/":
            # 根据模型类型添加默认路径
            if model_type == "openai-chat-completions":
                # 对于 chat completions 模型，添加 /v1/chat/completions
                if base_url.endswith("/"):
                    base_url = base_url.rstrip("/") + "/v1/chat/completions"
                else:
                    base_url = base_url + "/v1/chat/completions"
            elif model_type == "openai-completions":
                # 对于 completions 模型，如果用户没有提供路径，添加 /v1/completions
                # 但如果用户已经提供了包含 /v1/completions 的完整 URL（在前端提示中是这样），则不重复添加
                # 这里只处理纯域名的情况
                if base_url.endswith("/"):
                    base_url = base_url.rstrip("/") + "/v1/completions"
                else:
                    base_url = base_url + "/v1/completions"
        
        # 移除自动添加逻辑，因为前端现在提示用户输入完整 URL
        # 如果 base_url 已经是完整的（包含 /v1/completions），则直接使用
        model_args["base_url"] = base_url
    
    if model.get("max_concurrent"):
        model_args["num_concurrent"] = model["max_concurrent"]
        
    if model.get("max_tokens"):
        model_args["max_length"] = model["max_tokens"]
        
    if model.get("api_key"):
        model_args["api_key"] = model["api_key"]
    
    # 合并其他配置
    if model.get("other_config"):
        model_args.update(model["other_config"])
        
    return model_args


@router.post("/", response_model=ModelResponse)
async def create_model(request: ModelCreateRequest):
    """创建新的模型配置"""
    try:
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"创建模型失败: {str(e)}")


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


class ModelTestRequest(BaseModel):
    """测试模型连接请求"""
    model_config = {"protected_namespaces": ()}
    
    model_type: str
    base_url: Optional[str] = None  # 完整的 API URL
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    model_id: Optional[str] = None  # 如果提供 model_id，将从数据库获取真实的 api_key


@router.post("/test-connection")
async def test_model_connection(request: ModelTestRequest):
    """测试模型 API 连接，结合模型类型和模型标识进行测试"""
    import requests
    import json
    
    try:
        # base_url 应该已经是完整的 URL（包含端口）
        if not request.base_url:
            raise HTTPException(status_code=400, detail="base_url 是必需的")
        
        test_url = request.base_url.strip()
        
        # 如果提供了 model_id，从数据库获取真实的 api_key
        api_key = request.api_key
        if request.model_id:
            if request.model_id in models_db:
                model_data = models_db[request.model_id]
            else:
                model_data = load_model_from_file(request.model_id)
                if model_data:
                    models_db[request.model_id] = model_data
                else:
                    raise HTTPException(status_code=404, detail="模型不存在")
            
            # 使用数据库中的真实 api_key（如果存在）
            if model_data.get("api_key"):
                api_key = model_data["api_key"]
        
        # 根据模型类型发送不同的测试请求
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        if request.model_type in ["openai-chat-completions", "openai-completions"]:
            # OpenAI 兼容 API - 需要模型标识进行测试
            if not request.model_name:
                return {
                    "success": False,
                    "message": "缺少模型标识",
                    "details": f"模型类型 {request.model_type} 需要提供模型标识（model_name）才能进行测试"
                }
            
            # 首先尝试调用 /models 端点，验证 API 可访问性并检查模型是否存在
            models_url = test_url.rstrip('/')
            if not models_url.endswith('/models'):
                # 如果 URL 不包含 /models，尝试添加
                if '/v1' in models_url:
                    models_url = models_url.rstrip('/') + '/models'
                else:
                    models_url = models_url.rstrip('/') + '/v1/models'
            
            available_models = []
            try:
                response = requests.get(models_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    # 支持多种响应格式：
                    # 1. OpenAI 格式: {"data": [{"id": "model1"}, ...]}
                    # 2. 直接列表格式: [{"id": "model1"}, ...]
                    # 3. Ollama 等格式: {"models": [{"name": "model1"}, ...]}
                    if "data" in models_data:
                        available_models = [m.get("id", "") or m.get("name", "") for m in models_data.get("data", [])]
                    elif isinstance(models_data, list):
                        available_models = [m.get("id", "") or m.get("name", "") for m in models_data]
                    elif "models" in models_data:
                        available_models = [m.get("id", "") or m.get("name", "") for m in models_data.get("models", [])]
            except Exception:
                # 如果 /v1/models 失败，可能是 Ollama 服务，尝试原生 API
                # 尝试从 base_url 提取基础 URL（去掉 /v1 部分）
                try:
                    base_host = test_url.rstrip('/')
                    if '/v1' in base_host:
                        base_host = base_host[:base_host.rindex('/v1')]
                    ollama_tags_url = base_host.rstrip('/') + '/api/tags'
                    ollama_response = requests.get(ollama_tags_url, timeout=5)
                    if ollama_response.status_code == 200:
                        ollama_data = ollama_response.json()
                        if "models" in ollama_data:
                            available_models = [m.get("name", "") for m in ollama_data.get("models", [])]
                except Exception:
                    pass
            
            # 构建 completions 端点 URL
            completions_url = test_url.rstrip('/')
            # 检查 URL 是否已经包含完整的端点路径
            if not (completions_url.endswith('/chat/completions') or completions_url.endswith('/completions')):
                # 如果 URL 以 /v1 结尾，说明是基础 URL，需要添加端点路径
                if completions_url.endswith('/v1'):
                    if request.model_type == "openai-chat-completions":
                        completions_url = completions_url + '/chat/completions'
                    else:
                        completions_url = completions_url + '/completions'
                # 如果 URL 包含 /v1 但不以 /v1 结尾，可能是其他路径
                elif '/v1' in completions_url:
                    # URL 已经包含 /v1，可能不需要再添加
                    if request.model_type == "openai-chat-completions":
                        if not completions_url.endswith('/chat/completions'):
                            completions_url = completions_url.rstrip('/') + '/chat/completions'
                    else:
                        if not completions_url.endswith('/completions'):
                            completions_url = completions_url.rstrip('/') + '/completions'
                # 如果 URL 不包含 /v1，添加完整的路径
                else:
                    if request.model_type == "openai-chat-completions":
                        completions_url = completions_url.rstrip('/') + '/v1/chat/completions'
                    else:
                        completions_url = completions_url.rstrip('/') + '/v1/completions'
            
            # 根据模型类型构建不同的测试 payload，使用实际的模型名称
            if request.model_type == "openai-chat-completions":
                test_payload = {
                    "model": request.model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5
                }
            else:
                test_payload = {
                    "model": request.model_name,
                    "prompt": "Hello",
                    "max_tokens": 5
                }
            
            try:
                response = requests.post(
                    completions_url,
                    headers={**headers, "Content-Type": "application/json"},
                    json=test_payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    # 成功获取响应
                    response_data = response.json()
                    details = f"成功连接到 API，模型 '{request.model_name}' 可用"
                    if available_models:
                        if request.model_name in available_models:
                            details += f"，该模型在可用模型列表中"
                        else:
                            details += f"（注意：该模型不在 /models 端点返回的列表中，但 API 调用成功）"
                    return {
                        "success": True,
                        "message": "连接测试成功",
                        "details": details
                    }
                elif response.status_code == 400:
                    # 400 可能是参数错误，但说明 API 可访问
                    error_msg = response.text[:200]
                    return {
                        "success": False,
                        "message": "API 可访问但请求参数有误",
                        "details": f"模型 '{request.model_name}' 可能不存在或参数不正确。错误: {error_msg}"
                    }
                elif response.status_code == 401:
                    return {
                        "success": False,
                        "message": "认证失败",
                        "details": "API Key 无效或已过期，请检查 API Key 配置"
                    }
                elif response.status_code == 404:
                    # 404 错误，提供更详细的诊断信息
                    error_msg = response.text[:500] if response.text else ""
                    details = f"模型 '{request.model_name}' 不存在或无法访问"
                    
                    # 尝试提供有用的诊断信息
                    if available_models:
                        details += f"。可用模型列表: {', '.join(available_models[:10])}"
                        if len(available_models) > 10:
                            details += f" 等共 {len(available_models)} 个模型"
                        details += "。请检查模型标识是否正确，或者模型是否已正确加载。"
                    else:
                        details += "。无法获取可用模型列表，可能是 API 端点格式不同或需要特殊配置。"
                        details += " 请确认：1) 模型名称是否正确 2) 模型是否已在服务端加载 3) API URL 是否正确"
                    
                    if error_msg:
                        details += f" 错误详情: {error_msg}"
                    
                    return {
                        "success": False,
                        "message": "模型不存在",
                        "details": details
                    }
                else:
                    return {
                        "success": False,
                        "message": "连接失败",
                        "details": f"API 返回状态码: {response.status_code}，响应: {response.text[:200]}"
                    }
            except requests.exceptions.Timeout:
                return {
                    "success": False,
                    "message": "连接超时",
                    "details": "请求超时，请检查 URL 和网络连接"
                }
            except requests.exceptions.ConnectionError:
                return {
                    "success": False,
                    "message": "连接失败",
                    "details": "无法连接到服务器，请检查 URL 和端口是否正确"
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": "连接失败",
                    "details": f"错误: {str(e)}"
                }
        
        elif request.model_type == "hf":
            # HuggingFace 模型 - 本地模型，主要检查配置
            if not request.model_name:
                return {
                    "success": False,
                    "message": "缺少模型标识",
                    "details": "HuggingFace 模型需要提供模型标识（model_name）"
                }
            return {
                "success": True,
                "message": "配置检查通过",
                "details": f"HuggingFace 模型 '{request.model_name}' 配置正确（注意：此测试仅验证配置，实际模型加载将在评测时进行）"
            }
        
        else:
            return {
                "success": False,
                "message": "不支持的模型类型",
                "details": f"模型类型 {request.model_type} 暂不支持连接测试"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "message": "测试失败",
            "details": str(e)
        }


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str):
    """获取单个模型详情"""
    # 如果 model_id 包含 '/'，说明可能是路由匹配错误，应该匹配到更具体的路由
    if '/' in model_id:
        raise HTTPException(status_code=404, detail="模型不存在")
    
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
    try:
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
            # 如果传入的是 "***"，表示不更新，保持原有值
            if update_data["api_key"] == "***":
                del update_data["api_key"]
            # 如果传入的是空字符串，表示清除 api_key（设置为 None）
            elif update_data["api_key"] == "":
                model_data["api_key"] = None
                del update_data["api_key"]  # 已经从 model_data 中更新，不需要再在 update_data 中处理
            # 否则更新为新值（包括非空字符串）
        
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"更新模型失败: {str(e)}")


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """删除模型配置"""
    if model_id not in models_db:
        model_data = load_model_from_file(model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="模型不存在")
    
    # 删除模型文件
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
                "description": "OpenAI 兼容的聊天完成 API（支持 Ollama、vLLM、OpenAI 等）",
                "requires": ["base_url", "model_name"],
                "optional": ["api_key", "max_concurrent", "max_tokens"]
            },
            {
                "value": "openai-completions",
                "label": "OpenAI Completions",
                "description": "OpenAI 兼容的文本完成 API",
                "requires": ["base_url", "model_name"],
                "optional": ["api_key", "max_concurrent", "max_tokens"]
            },
            {
                "value": "hf",
                "label": "HuggingFace",
                "description": "HuggingFace Transformers 本地模型",
                "requires": ["model_name"],
                "optional": ["port", "max_tokens"]
            }
        ]
    }

