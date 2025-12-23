#!/bin/bash
# 启动后端服务

cd "$(dirname "$0")"

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

# 检查并安装依赖
echo ""
echo "正在检查依赖..."
cd web_backend
if [ -f "requirements.txt" ]; then
    echo "发现 requirements.txt，正在检查依赖..."
    python -c "import fastapi" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "fastapi 未安装，正在安装依赖..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "错误: 依赖安装失败，请手动运行: pip install -r web_backend/requirements.txt"
            exit 1
        fi
    else
        echo "依赖检查通过"
    fi
else
    echo "警告: 未找到 requirements.txt"
fi

echo ""
echo "正在启动后端服务..."
echo "API 文档地址: http://localhost:8087/docs"
echo ""

python main.py
