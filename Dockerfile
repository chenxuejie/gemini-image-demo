# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 镜像元数据
LABEL maintainer="your-email@example.com"
LABEL description="Gemini + InsightFace 视频分析平台"
LABEL version="1.0.0"

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

# 设置工作目录
WORKDIR /app

# 安装系统依赖和构建工具
RUN apt-get update && apt-get install -y \
    # 构建工具（insightface 编译需要）
    build-essential \
    gcc \
    g++ \
    make \
    # OpenCV 依赖
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # 视频处理依赖
    ffmpeg \
    # 中文字体支持
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    fonts-noto-cjk \
    # 网络工具
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && g++ --version

# 安装 Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .
COPY insightface_utils.py .
COPY index.html .
COPY analyze_video_direct.py .

# 创建必要的目录
RUN mkdir -p face_db/embeddings labeled_videos

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# 启动命令
CMD ["python", "app.py"]
