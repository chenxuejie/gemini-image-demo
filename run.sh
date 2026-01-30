#!/bin/bash

# Gemini 图片推理 Demo 启动脚本

echo "=========================================="
echo "  Gemini 2.5 Flash Lite 图片推理 Demo"
echo "=========================================="

# 进入脚本所在目录
cd "$(dirname "$0")"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 检查并安装依赖
echo ""
echo "检查依赖..."
pip install -r requirements.txt -q

# 检查 gcloud 认证
echo ""
echo "检查 Google Cloud 认证..."
if ! gcloud auth application-default print-access-token &> /dev/null; then
    echo "警告: 未检测到 Google Cloud 认证"
    echo "请运行: gcloud auth application-default login"
    echo ""
fi

# 启动服务
echo ""
echo "启动服务..."
echo "访问地址: http://localhost:8080"
echo ""
echo "按 Ctrl+C 停止服务"
echo "=========================================="
echo ""

python app.py
