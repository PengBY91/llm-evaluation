export OPENAI_API_KEY=sk-wFHN2ySjUYxCx3LrWAkJEMB11FMxYDvF6DHdye9yVDwIH2no
conda run -n lm-eval lm_eval --model openai-chat-completions \
    --model_args model=deepseek-chat,base_url=https://api.huiyan-ai.cn/v1/chat/completions,num_concurrent=10,tokenizer=cl100k_tiktoken,tokenizer_backend=tiktoken \
    --tasks mmlu \
    --apply_chat_template \
    --limit 20