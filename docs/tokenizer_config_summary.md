# LM-Eval Tokenizer 配置总结

根据 `lm_eval/models/` 目录中各种模型的实现，以下是 tokenizer 设置的通用模式：

## 1. HuggingFace 模型 (`huggingface.py`)

**Tokenizer 加载逻辑：**
```python
if tokenizer:
    # 如果提供了 tokenizer 参数
    if isinstance(tokenizer, str):
        # 字符串：可以是 HuggingFace 模型名称或本地路径
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, **kwargs)
    else:
        # PreTrainedTokenizer 对象
        self.tokenizer = tokenizer
else:
    # 如果没有提供 tokenizer，使用 pretrained 模型名称
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained, **kwargs)
```

**关键点：**
- `tokenizer` 参数可以是：
  - HuggingFace 模型名称（如 `"Qwen/Qwen2.5-8B"`）
  - 本地路径（如 `"/path/to/tokenizer"` 或 `"/path/to/model"`）
  - PreTrainedTokenizer 对象（直接传入已加载的 tokenizer）
- 如果未提供 `tokenizer`，默认使用 `pretrained` 参数的值

## 2. API 模型 (`api_models.py` - TemplateAPI)

**三种 tokenizer_backend：**

### a) `tokenizer_backend="huggingface"` (默认)
```python
if self.tokenizer_backend == "huggingface":
    import transformers
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.tokenizer if self.tokenizer else self.model,  # 关键：如果有 tokenizer 参数就用它，否则用 model
        trust_remote_code=trust_remote_code,
        revision=revision,
        use_fast=use_fast_tokenizer,
    )
```

**关键点：**
- 如果 `self.tokenizer` 有值（通过 `tokenizer` 参数传入），使用它
- 否则使用 `self.model`（即模型名称）
- **这意味着如果模型名称包含特殊字符（如 `:`），无法直接作为 HuggingFace repo id，必须通过 `tokenizer` 参数显式指定**

### b) `tokenizer_backend="tiktoken"`
```python
elif self.tokenizer_backend == "tiktoken":
    import tiktoken
    self.tokenizer = tiktoken.encoding_for_model(self.model)
```
- 只支持标准的 OpenAI 模型名称（如 `gpt-3.5-turbo`, `text-davinci-003` 等）
- 不支持包含特殊字符的模型名称

### c) `tokenizer_backend="remote"`
```python
elif self.tokenizer_backend == "remote":
    from lm_eval.utils import RemoteTokenizer
    self.tokenizer = RemoteTokenizer(
        self.base_url,
        self.timeout,
        self.verify_certificate,
        self.ca_cert_path,
        self.auth_token,
    )
```
- 使用服务器的 tokenizer endpoints（如 vLLM 服务器）
- 不需要本地安装 transformers 或指定 tokenizer 路径

## 3. vLLM 模型 (`vllm_causallms.py`)

```python
from vllm.transformers_utils.tokenizer import get_tokenizer

self.tokenizer = get_tokenizer(
    tokenizer if tokenizer else pretrained,  # 如果提供了 tokenizer 就用它，否则用 pretrained
    tokenizer_mode=tokenizer_mode,
    trust_remote_code=trust_remote_code,
    revision=tokenizer_revision,
)
```

## 4. 其他模型模式

- **Mamba 模型**: 默认使用 `"EleutherAI/gpt-neox-20b"` 作为 tokenizer，但可以通过 `tokenizer` 参数覆盖
- **NeMo 模型**: 直接使用模型自带的 tokenizer (`self.tokenizer = self.model.tokenizer`)
- **SGLang 模型**: 使用 SGLang 的 tokenizer manager

## 总结：对于本地 LLM（如 `qwen3:8b`）的配置

### 问题
模型名称 `qwen3:8b` 包含 `:` 字符，无法直接作为 HuggingFace repo id 使用。

### 解决方案

#### 方案 1：在 `other_config` 中指定 tokenizer（推荐）
```json
{
  "tokenizer": "Qwen/Qwen2.5-7B",
  "tokenizer_backend": "huggingface"
}
```

**注意**：Qwen 系列没有标准的 8B 模型，只有 7B、14B、32B、72B 等。如果你的模型是 8B（可能是量化或自定义版本），建议使用 7B 的 tokenizer。

可用的 Qwen 模型：
- Qwen2.5: `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B`, `Qwen/Qwen2.5-72B`
- Qwen2: `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-1.5B`, `Qwen/Qwen2-7B`, `Qwen/Qwen2-72B`

或使用本地路径：
```json
{
  "tokenizer": "/path/to/local/tokenizer",
  "tokenizer_backend": "huggingface"
}
```

#### 方案 2：使用 remote tokenizer（如果服务器支持）
```json
{
  "tokenizer_backend": "remote"
}
```
这样会使用服务器提供的 tokenizer，不需要本地配置。

### 代码中的自动处理逻辑

当前代码（`web_backend/api/models.py`）会自动：
1. 检测模型名称是否包含特殊字符（如 `:` 或 `/`）
2. 如果包含，自动设置 `tokenizer_backend="huggingface"`
3. 但是**仍然需要用户手动指定 `tokenizer` 参数**，因为模型名称无法直接使用

### 最佳实践

对于包含特殊字符的模型名称（如 `qwen3:8b`）：
1. 在 `other_config` 中明确指定 `tokenizer` 参数
2. 使用对应的 HuggingFace 模型名称（如 `Qwen/Qwen2.5-8B`）或本地 tokenizer 路径
3. 设置 `tokenizer_backend="huggingface"`（代码会自动设置，但可以显式指定）
