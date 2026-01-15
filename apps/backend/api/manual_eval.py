"""
手动评测 API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
import os
import requests
from pathlib import Path
from api.models import models_db, load_model_from_file

router = APIRouter()

# 手动评测结果存储目录
MANUAL_EVALS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "manual_evals"
MANUAL_EVALS_DIR.mkdir(parents=True, exist_ok=True)

class GenerateRequest(BaseModel):
    model_ids: List[str]
    system_prompt: str = "You are a helpful assistant."
    user_prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    n: int = 1  # 每个模型生成的回答数量

class EvaluationItem(BaseModel):
    model_id: str
    model_name: str
    answer: str
    score: Optional[int] = None
    rank: Optional[int] = None

class SaveEvalRequest(BaseModel):
    name: str
    system_prompt: str
    user_prompt: str
    evaluations: List[EvaluationItem]
    ai_evaluation: Optional[str] = None

@router.post("/generate")
async def generate_answers(request: GenerateRequest):
    """为选定的模型生成答案"""
    results = []
    
    for model_id in request.model_ids:
        model_data = models_db.get(model_id) or load_model_from_file(model_id)
        if not model_data:
            results.append({
                "model_id": model_id,
                "error": "模型未找到"
            })
            continue
            
        # 尝试为该模型生成 n 次回答
        for i in range(max(1, min(request.n, 5))):
            try:
                print(f"DEBUG: Generating answer {i+1}/{request.n} for model {model_id} ({model_data.get('name')})")
                answer = ""
                backend_type = model_data.get("backend_type", "")
                if backend_type == "openai-api":
                    # Use requests to call OpenAI-compatible API
                    url = model_data["base_url"].strip()
                    
                    # Handle endpoint path - always use chat/completions for generation
                    if not url.endswith("/chat/completions"):
                        url = url.rstrip("/") + ("/chat/completions" if "/v1" in url else "/v1/chat/completions")
                    
                    payload = {
                        "model": model_data["model_name"],
                        "messages": [
                            {"role": "system", "content": request.system_prompt},
                            {"role": "user", "content": request.user_prompt}
                        ],
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "n": 1
                    }

                    headers = {"Content-Type": "application/json"}
                    if model_data.get("api_key"):
                        headers["Authorization"] = f"Bearer {model_data['api_key']}"
                    
                    response = requests.post(url, json=payload, headers=headers, timeout=120)
                    if response.status_code == 200:
                        resp_json = response.json()
                        answer = resp_json["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"API error: {response.status_code} {response.text}")
                else:
                    raise Exception(f"Unsupported backend type: {backend_type}")
                    
                results.append({
                    "model_id": model_id,
                    "model_name": model_data.get("name", "Unknown") + (f" (#{i+1})" if request.n > 1 else ""),
                    "answer": answer
                })
                print(f"DEBUG: Successfully generated answer {i+1} for model {model_id}")
            except Exception as e:
                results.append({
                    "model_id": model_id,
                    "model_name": model_data.get("name", "Unknown") + (f" (#{i+1})" if request.n > 1 else ""),
                    "error": str(e)
                })
                print(f"ERROR: Failed to generate answer {i+1} for model {model_id}: {str(e)}")
            
    return {"results": results}

@router.post("/evaluate")
async def ai_evaluate(request: Dict[str, Any]):
    """使用 AI 模型对答案进行评价"""
    evaluator_model_id = request.get("evaluator_model_id")
    system_prompt = request.get("system_prompt")
    user_prompt = request.get("user_prompt")
    answers = request.get("answers", []) # List of {model_name, answer}
    
    if not evaluator_model_id:
        raise HTTPException(status_code=400, detail="请选择评分模型")
        
    model_data = models_db.get(evaluator_model_id) or load_model_from_file(evaluator_model_id)
    if not model_data:
        raise HTTPException(status_code=404, detail="评分模型未找到")
        
    # 构建评价提示词
    eval_prompt = f"请作为一名专家，评价以下多个模型对同一个问题的回答。\n\n"
    eval_prompt += f"【系统提示词】：{system_prompt}\n"
    eval_prompt += f"【用户问题】：{user_prompt}\n\n"
    
    for i, item in enumerate(answers):
        eval_prompt += f"--- 模型 {i+1} ({item['model_name']}) ---\n{item['answer']}\n\n"
        
    eval_prompt += "请对比上述回答，指出各自的优缺点，并给出一个综合排名建议。"
    
    try:
        # 复用 generate_answers 逻辑调用接口
        url = model_data["base_url"].strip()
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + ("/chat/completions" if "/v1" in url else "/v1/chat/completions")
            
        payload = {
            "model": model_data["model_name"],
            "messages": [
                {"role": "system", "content": "You are an expert evaluator of LLM responses."},
                {"role": "user", "content": eval_prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.3
        }
        
        headers = {"Content-Type": "application/json"}
        if model_data.get("api_key"):
            headers["Authorization"] = f"Bearer {model_data['api_key']}"
            
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        if response.status_code == 200:
            return {"evaluation": response.json()["choices"][0]["message"]["content"]}
        else:
            raise Exception(f"AI 评分出错: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save")
async def save_eval(request: SaveEvalRequest):
    """保存评测结果"""
    eval_id = str(uuid.uuid4())
    filename = f"{eval_id}.json"
    
    data = {
        "id": eval_id,
        "created_at": datetime.now().isoformat(),
        **request.dict()
    }
    
    with open(MANUAL_EVALS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return {"message": "保存成功", "id": eval_id}

@router.get("/list")
async def list_evals():
    """获取手动评测记录列表"""
    results = []
    for file in MANUAL_EVALS_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append({
                    "id": data["id"],
                    "name": data["name"],
                    "created_at": data["created_at"],
                    "model_count": len(data["evaluations"]),
                    "user_prompt": data["user_prompt"][:50] + "..." if len(data["user_prompt"]) > 50 else data["user_prompt"]
                })
        except:
            continue
            
    # 按时间降序
    try:
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    except Exception as e:
        print(f"排序评测记录失败: {e}")
        
    return results

@router.get("/{eval_id}")
async def get_eval(eval_id: str):
    """获取单个评测详情"""
    file = MANUAL_EVALS_DIR / f"{eval_id}.json"
    if not file.exists():
        raise HTTPException(status_code=404, detail="记录不存在")
        
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

@router.delete("/{eval_id}")
async def delete_eval(eval_id: str):
    """删除评测记录"""
    file = MANUAL_EVALS_DIR / f"{eval_id}.json"
    if file.exists():
        file.unlink()
        return {"message": "已删除"}
    raise HTTPException(status_code=404, detail="记录不存在")
