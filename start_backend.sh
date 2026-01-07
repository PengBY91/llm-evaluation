#!/bin/bash
# 启动后端服务
set -x # Enable debug mode to see exactly where it hangs

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
cd apps/backend
if [ -f "requirements.txt" ]; then
    echo "发现 requirements.txt，正在检查依赖..."
    python -c "import fastapi" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "fastapi 未安装，正在尝试安装依赖 (这可能由于网络原因卡住)..."
        # 增加超时或重试逻辑
        pip install -v -r requirements.txt --timeout 30
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
echo "正在清理旧的后端进程 (端口 8087)..."
# 自动杀死占用 8087 端口的进程
if command -v fuser &> /dev/null; then
    fuser -k 8087/tcp &> /dev/null
else
    PID=$(lsof -ti:8087)
    if [ ! -z "$PID" ]; then
        kill -9 $PID &> /dev/null
    fi
fi
sleep 1

echo "正在启动后端服务..."
echo "API 文档地址: http://localhost:8087/docs"
echo ""

# 确保代理环境变量被传递给 Python 进程
# 这对于使用 aiohttp 的 lm-eval 库非常重要
if [ ! -z "$HTTP_PROXY" ] || [ ! -z "$http_proxy" ] || [ ! -z "$HTTPS_PROXY" ] || [ ! -z "$https_proxy" ]; then
    echo "检测到代理配置，将传递给后端进程"
    export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
    export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
    export http_proxy="${http_proxy:-$HTTP_PROXY}"
    export https_proxy="${https_proxy:-$HTTPS_PROXY}"
fi

python main.py

