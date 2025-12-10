#!/bin/bash
# 启动前端应用

cd "$(dirname "$0")/web_frontend"

# 检查是否已安装依赖
if [ ! -d "node_modules" ]; then
    echo "检测到未安装依赖，正在安装..."
    npm install
    if [ $? -ne 0 ]; then
        echo "依赖安装失败，请检查网络连接和 Node.js 环境"
        exit 1
    fi
    echo "依赖安装完成！"
    echo ""
fi

echo "正在启动前端应用..."
echo "前端地址: http://localhost:3003"
echo ""

npm run dev

