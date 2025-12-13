# Ollama API 与 Loglikelihood 支持指南

## 问题说明

### Ollama API 不支持 logprobs
根据 [Ollama GitHub Issue #2415](https://github.com/ollama/ollama/issues/2415)，Ollama API **目前不支持 `logprobs` 参数**。这意味着：

1. **无法获取 loglikelihoods**：Ollama 的 `/v1/completions` 和 `/v1/chat/completions` 端点都不支持返回 token 级别的 log probabilities
2. **无法运行需要 loglikelihood 的任务**：如 `multiple_choice` 类型任务（如 MMLU、ARC 等）

### 当前限制
- `openai-completions` 类型有硬编码限制：只允许 `babbage-002` 和 `davinci-002` 使用 loglikelihood
- 即使用户使用本地 Ollama 服务器，如果 API 不支持 logprobs，也无法获取 loglikelihoods

## 解决方案

### 方案 1：使用 `generate_until` 类型的任务（推荐）

对于 Ollama 模型，应该使用 **生成式任务**而不是选择题任务：

**可选的任务类型：**
- `mmlu_generative` 而不是 `mmlu`（multiple_choice）
- `hellaswag_generative` 而不是 `hellaswag`（multiple_choice）
- 其他 `generate_until` 输出类型的任务

**优势：**
- 不需要 logprobs
- Ollama 完全支持
- 可以评估模型的生成能力

### 方案 2：使用支持 logprobs 的本地服务器

如果你想运行需要 loglikelihood 的任务，可以考虑使用支持 logprobs 的本地服务器：

#### a) vLLM 服务器
vLLM 支持 logprobs：
```bash
# 启动 vLLM 服务器（支持 logprobs）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B \
    --port 8000
```

然后配置：
- `base_url`: `http://localhost:8000/v1`
- `model_type`: `openai-completions` 或 `openai-chat-completions`

#### b) 其他支持 OpenAI API 格式的本地服务器
任何实现了 OpenAI API 格式并支持 `logprobs` 参数的服务器都可以。

### 方案 3：使用 HuggingFace 本地模型

如果需要在本地运行 loglikelihood 任务，可以使用 HuggingFace 模型类型：

```json
{
  "model_type": "hf",
  "model_name": "/path/to/local/model",
  ...
}
```

这样可以完全支持 loglikelihood 任务。

## 代码中的处理

### 当前限制的位置
- `lm_eval/models/openai_completions.py` 第 281-288 行：`OpenAICompletionsAPI.loglikelihood()` 方法中有硬编码的限制
- 这个限制是针对 OpenAI 官方 API 的，但也会影响使用 `openai-completions` 类型的本地服务器

### 为什么有这个限制？
OpenAI 官方 API 只有 `babbage-002` 和 `davinci-002` 两个模型支持 logprobs，其他模型（如 gpt-3.5-turbo）都不支持。

### 对于本地服务器的建议

如果你使用的是**本地服务器**（如 Ollama、vLLM 等），应该：

1. **检查服务器是否支持 logprobs**：
   - Ollama：不支持 ❌
   - vLLM：支持 ✅
   - 其他服务器：需要查看其文档

2. **如果不支持 logprobs**：
   - 使用 `generate_until` 类型的任务
   - 或者切换到支持 logprobs 的服务器

3. **如果服务器支持 logprobs**：
   - 代码中的限制可能会阻止你使用，但这是 OpenAI 官方 API 的限制
   - 可以考虑直接使用 `local-completions` 类型（如果框架支持）

## 推荐配置

### 对于 Ollama（不支持 logprobs）
```json
{
  "model_type": "openai-completions",
  "base_url": "http://localhost:11434/v1/completions",
  "model_name": "qwen3:8b-q4_K_M",
  "other_config": {
    "tokenizer": "Qwen/Qwen2.5-7B",
    "tokenizer_backend": "huggingface"
  }
}
```

**建议的任务类型：**
- 使用 `generate_until` 类型的任务
- 避免 `multiple_choice` 类型的任务（需要 loglikelihood）

### 对于 vLLM（支持 logprobs）
```json
{
  "model_type": "openai-completions",
  "base_url": "http://localhost:8000/v1/completions",
  "model_name": "Qwen/Qwen2.5-7B",
  "other_config": {
    "tokenizer_backend": "remote"  // vLLM 支持 remote tokenizer
  }
}
```

**可以运行所有类型的任务**，包括需要 loglikelihood 的任务。

## 总结

- **Ollama 不支持 logprobs**，因此无法运行需要 loglikelihood 的任务
- **使用 `generate_until` 类型的任务**可以绕过这个限制
- 如果需要 loglikelihood，考虑使用 vLLM 或其他支持 logprobs 的服务器
