# `/data` 目录下任务输出类型总结

根据任务配置文件的 `output_type` 字段，以下是 `/data` 目录下各任务的分类：

## ✅ 不需要 loglikelihood 的任务（`generate_until` 类型）

这些任务可以在不支持 logprobs 的模型（如 Ollama）上运行：

### 数学和推理
- **gsm8k** - `output_type: generate_until` ✅
  - 数学文字问题，生成答案
- **hendrycks_math** - `output_type: generate_until` ✅
  - 数学问题
- **agieval/math** - `output_type: generate_until` ✅
- **agieval/gaokao-mathcloze** - `output_type: generate_until` ✅

### 代码生成
- **humaneval** - `output_type: generate_until` ✅
  - 代码生成任务

### 问答
- **triviaqa** - `output_type: generate_until` ✅
  - 开放域问答

### 真实性
- **truthfulqa_gen** - `output_type: generate_until` ✅
  - 真实性问题生成版本

### MMLU 生成版本
- **mmlu/generative** - `output_type: generate_until` ✅
- **mmlu/flan_n_shot/generative** - `output_type: generate_until` ✅
- **mmlu/flan_cot_fewshot** - `output_type: generate_until` ✅
- **mmlu/flan_cot_zeroshot** - `output_type: generate_until` ✅

### ARC 生成版本
- **arc_challenge_chat** - `output_type: generate_until` ✅

### 其他生成任务
- **longbench/** 目录下的任务（大多数是 `generate_until`）

## ❌ 需要 loglikelihood 的任务

这些任务需要 logprobs 支持，无法在不支持 logprobs 的模型（如 Ollama）上运行：

### Multiple Choice 类型（需要 loglikelihood）

- **arc_easy** - `output_type: multiple_choice` ❌
- **arc_challenge** - `output_type: multiple_choice` ❌（注意：有 `arc_challenge_chat` 是 generate_until）
- **hellaswag** - `output_type: multiple_choice` ❌
- **piqa** - `output_type: multiple_choice` ❌
- **winogrande** - `output_type: multiple_choice` ❌
- **mmlu/default** - `output_type: multiple_choice` ❌
- **mmlu/continuation** - `output_type: multiple_choice` ❌
- **agieval/aqua-rat** - `output_type: multiple_choice` ❌
- **openbookqa** - `output_type: multiple_choice` ❌
- **super_glue/** 下的许多任务 - `output_type: multiple_choice` ❌
- **bbh/** 下的许多任务 - `output_type: multiple_choice` ❌
- **mmlu_pro** - `output_type: multiple_choice` ❌
- **ceval** - `output_type: multiple_choice` ❌

### Loglikelihood Rolling 类型（需要 loglikelihood）

- **wikitext** - `output_type: loglikelihood_rolling` ❌
  - 用于计算 perplexity

- **lambada** - `output_type: loglikelihood` 或 `multiple_choice` ❌

## 📋 快速参考表

| 任务名称 | Output Type | 是否需要 loglikelihood | Ollama 支持 |
|---------|-------------|---------------------|------------|
| gsm8k | generate_until | ❌ 不需要 | ✅ 支持 |
| humaneval | generate_until | ❌ 不需要 | ✅ 支持 |
| truthfulqa_gen | generate_until | ❌ 不需要 | ✅ 支持 |
| triviaqa | generate_until | ❌ 不需要 | ✅ 支持 |
| hendrycks_math | generate_until | ❌ 不需要 | ✅ 支持 |
| mmlu/generative | generate_until | ❌ 不需要 | ✅ 支持 |
| arc_challenge_chat | generate_until | ❌ 不需要 | ✅ 支持 |
| arc_easy | multiple_choice | ✅ 需要 | ❌ 不支持 |
| arc_challenge | multiple_choice | ✅ 需要 | ❌ 不支持 |
| hellaswag | multiple_choice | ✅ 需要 | ❌ 不支持 |
| piqa | multiple_choice | ✅ 需要 | ❌ 不支持 |
| winogrande | multiple_choice | ✅ 需要 | ❌ 不支持 |
| mmlu/default | multiple_choice | ✅ 需要 | ❌ 不支持 |
| openbookqa | multiple_choice | ✅ 需要 | ❌ 不支持 |
| wikitext | loglikelihood_rolling | ✅ 需要 | ❌ 不支持 |
| lambada | loglikelihood/multiple_choice | ✅ 需要 | ❌ 不支持 |

## 💡 建议

对于使用 Ollama 等不支持 logprobs 的模型：

1. **优先使用 `generate_until` 类型的任务**：
   - `gsm8k` - 数学问题
   - `humaneval` - 代码生成
   - `truthfulqa_gen` - 真实性生成
   - `triviaqa` - 开放域问答
   - `hendrycks_math` - 数学
   - `mmlu/generative` - MMLU 生成版本

2. **避免使用 `multiple_choice` 类型的任务**：
   - `arc_easy`, `arc_challenge`
   - `hellaswag`
   - `piqa`
   - `winogrande`
   - `mmlu/default`
   - `openbookqa`

3. **使用生成版本替代选择题版本**：
   - 使用 `mmlu/generative` 而不是 `mmlu/default`
   - 使用 `arc_challenge_chat` 而不是 `arc_challenge`

## 如何查找任务的 output_type

你可以通过以下方式查找任务的 output_type：

```bash
# 查找任务的配置文件
find lm_eval/tasks -name "*.yaml" | grep <task_name>

# 查看配置文件中的 output_type
grep "output_type:" lm_eval/tasks/<task_name>/*.yaml
```

或者通过代码：
```python
from lm_eval.tasks import TaskManager
task_manager = TaskManager()
task_info = task_manager.task_index.get("task_name", {})
yaml_path = task_info.get("yaml_path")
# 读取 YAML 文件查看 output_type
```
