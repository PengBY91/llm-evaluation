export OPENAI_API_KEY=sk-wFHN2ySjUYxCx3LrWAkJEMB11FMxYDvF6DHdye9yVDwIH2no
lm_eval --model openai-chat-completions \
    --model_args model=deepseek-chat,base_url=https://api.huiyan-ai.cn/v1/chat/completions,num_concurrent=10 \
    --tasks gsm8k \
    --apply_chat_template \
    --limit 200