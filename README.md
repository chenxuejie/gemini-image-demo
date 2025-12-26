# Gemini 2.5 Flash Lite 图片推理 Demo

这是一个简单的网页应用，用于上传图片并使用 Gemini 2.5 Flash Lite 模型进行推理分析。

## 功能特点

- 📁 支持拖拽或点击上传图片
- 🖼️ 自动调整图片尺寸以适应模型输入要求
- 🤖 使用 Gemini 2.5 Flash Lite 进行图片分析
- 📊 显示 Input Tokens、Output Tokens 和 Total Tokens

## 配置信息

- **Project ID**: cloud-llm-preview1
- **Model**: gemini-2.5-flash-lite
- **Location**: us-central1

## 快速开始

### 1. 安装依赖

```bash
cd gemini-image-demo
pip install -r requirements.txt
```

### 2. 配置 Google Cloud 认证

确保您已经配置了 Google Cloud 认证：

```bash
gcloud auth application-default login
gcloud config set project cloud-llm-preview1
```

### 3. 启动服务

```bash
python app.py
```

或使用启动脚本：

```bash
./run.sh
```

### 4. 访问应用

打开浏览器访问：http://localhost:8080

## 使用说明

1. 点击上传区域或拖拽图片到上传区域
2. 在提示词输入框中输入您想让 AI 分析的内容
3. 点击"提交分析"按钮
4. 等待分析完成，查看结果和 Token 使用情况

## API 接口

### POST /api/analyze

上传图片并进行分析。

**请求参数：**
- `image`: 图片文件 (multipart/form-data)
- `prompt`: 提示词 (string)

**响应示例：**
```json
{
    "response": "这是一张...",
    "input_tokens": 258,
    "output_tokens": 150,
    "total_tokens": 408,
    "image_info": {
        "original_size": "1920x1080",
        "processed_size": "1920x1080",
        "resized": false
    }
}
```

### GET /api/health

健康检查接口。

**响应示例：**
```json
{
    "status": "healthy",
    "project_id": "cloud-llm-preview1",
    "model_id": "gemini-2.5-flash-lite-preview-06-17",
    "location": "us-central1"
}
```

## 图片处理

- 支持格式：JPG, PNG, GIF, WebP
- 最大文件大小：20MB
- 如果图片尺寸超过 3072 像素，会自动等比例缩放

## 技术栈

- **前端**: HTML5, CSS3, JavaScript
- **后端**: Python Flask
- **AI**: Google Vertex AI (Gemini 2.5 Flash Lite)
- **图片处理**: Pillow
