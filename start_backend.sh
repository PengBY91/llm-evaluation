#!/bin/bash
# 启动后端服务

cd "$(dirname "$0")/web_backend"

# 激活 conda 环境
if command -v conda &> /dev/null; then
    echo "正在激活 conda 环境: lm-eval"
    eval "$(conda shell.bash hook)"
    conda activate lm-eval
    if [ $? -ne 0 ]; then
        echo "警告: 无法激活 conda 环境 'lm-eval'，将使用当前 Python 环境"
    fi
else
    echo "警告: 未找到 conda，将使用当前 Python 环境"
fi

echo ""
echo "正在启动后端服务..."
echo "API 文档地址: http://localhost:8087/docs"
echo ""

python3 main.py

