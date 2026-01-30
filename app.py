"""
Gemini 2.5 Flash Lite 图片/视频推理后端服务
使用 Google GenAI SDK 调用 Gemini 模型
"""

import os
import io
import base64
import logging
import tempfile
import subprocess
from datetime import datetime
import json
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from PIL import Image
from google import genai
from google.genai import types
import time

# OpenCV is optional - only needed for frames mode
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) not available. Frame extraction mode will be disabled. Install with: pip install opencv-python-headless")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# LangChain imports for testing media_resolution
try:
    from langchain_google_vertexai import ChatVertexAI
    from langchain_core.messages import HumanMessage
    from pydantic import Field
    from typing import Optional
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with: pip install langchain-google-vertexai")


# 自定义 ChatVertexAI 类，支持 media_resolution 参数
if LANGCHAIN_AVAILABLE:
    class ChatVertexAIWithMediaResolution(ChatVertexAI):
        """
        继承 ChatVertexAI，添加 media_resolution 支持。
        
        通过在 invoke 时传递 media_resolution 参数来控制图片分辨率。
        根据源码分析，media_resolution 在 _allowed_beta_params 中，
        需要通过 invoke 的 kwargs 传递，而不是在初始化时设置。
        """
        
        # 默认 media_resolution
        default_media_resolution: Optional[str] = Field(default=None)
        
        def invoke(self, input, config=None, *, stop=None, **kwargs):
            """重写 invoke 方法，注入 media_resolution 参数"""
            # 如果没有传入 media_resolution，使用默认值
            # 需要使用正确的枚举值格式：MEDIA_RESOLUTION_LOW, MEDIA_RESOLUTION_MEDIUM, MEDIA_RESOLUTION_HIGH
            if 'media_resolution' not in kwargs and self.default_media_resolution:
                # 转换为正确的枚举值格式
                resolution_map = {
                    'low': 'MEDIA_RESOLUTION_LOW',
                    'medium': 'MEDIA_RESOLUTION_MEDIUM',
                    'high': 'MEDIA_RESOLUTION_HIGH',
                    'unspecified': 'MEDIA_RESOLUTION_UNSPECIFIED'
                }
                enum_value = resolution_map.get(self.default_media_resolution, self.default_media_resolution)
                kwargs['media_resolution'] = enum_value
                logger.info(f"[ChatVertexAIWithMediaResolution] 注入 media_resolution={enum_value}")
            
            return super().invoke(input, config, stop=stop, **kwargs)
        
        async def ainvoke(self, input, config=None, *, stop=None, **kwargs):
            """重写 ainvoke 方法，注入 media_resolution 参数"""
            if 'media_resolution' not in kwargs and self.default_media_resolution:
                kwargs['media_resolution'] = self.default_media_resolution
            
            return await super().ainvoke(input, config, stop=stop, **kwargs)

# 配置
PROJECT_ID = "cloud-llm-preview2"
LOCATION = "us-central1"
LOCATION_GLOBAL = "global"
DEFAULT_MODEL_ID = "gemini-2.5-flash-lite"

# 初始化 GenAI 客户端
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# 可用模型列表
AVAILABLE_MODELS = {
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.0-flash-lite-001": "Gemini 2.0 Flash Lite",
    "gemini-3-pro-preview": "Gemini 3 Pro Preview",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview"
}

# 图片最大尺寸限制
MAX_IMAGE_SIZE = 4096
MAX_FILE_SIZE_MB = 20

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 最大请求大小
CORS(app)


def resize_image(image_bytes: bytes, target_width: int = None, target_height: int = None) -> tuple[bytes, dict]:
    """
    根据用户指定的尺寸调整图片大小
    返回处理后的图片字节和图片信息
    """
    img = Image.open(io.BytesIO(image_bytes))
    original_size = img.size
    original_format = img.format or 'JPEG'
    
    width, height = img.size
    new_width, new_height = width, height
    resized = False
    
    if target_width is not None or target_height is not None:
        if target_width is not None and target_height is not None:
            new_width = min(target_width, MAX_IMAGE_SIZE)
            new_height = min(target_height, MAX_IMAGE_SIZE)
        elif target_width is not None:
            new_width = min(target_width, MAX_IMAGE_SIZE)
            new_height = int(height * (new_width / width))
        else:
            new_height = min(target_height, MAX_IMAGE_SIZE)
            new_width = int(width * (new_height / height))
        
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        if new_width != width or new_height != height:
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized = True
    
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    output = io.BytesIO()
    save_format = 'JPEG' if original_format.upper() in ['JPG', 'JPEG'] else original_format
    if save_format.upper() not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
        save_format = 'JPEG'
    
    img.save(output, format=save_format, quality=95)
    processed_bytes = output.getvalue()
    
    info = {
        'original_size': original_size,
        'processed_size': (new_width, new_height),
        'resized': resized,
        'format': save_format
    }
    
    return processed_bytes, info


# Media Resolution 映射
MEDIA_RESOLUTION_MAP = {
    'unspecified': types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
    'low': types.MediaResolution.MEDIA_RESOLUTION_LOW,
    'medium': types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    'high': types.MediaResolution.MEDIA_RESOLUTION_HIGH
}


def analyze_image_with_gemini(image_bytes: bytes, prompt: str, mime_type: str, model_id: str = None, media_resolution: str = None) -> dict:
    """使用 Gemini 模型分析图片"""
    global client
    
    try:
        model_id = model_id or DEFAULT_MODEL_ID
        
        if model_id in ["gemini-3-pro-preview", "gemini-3-flash-preview"]:
            current_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION_GLOBAL)
        else:
            current_client = client
        
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        
        config_params = {"max_output_tokens": 2048, "temperature": 0.4}
        
        if model_id == "gemini-2.5-flash-lite":
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        
        if media_resolution and media_resolution in MEDIA_RESOLUTION_MAP:
            config_params["media_resolution"] = MEDIA_RESOLUTION_MAP[media_resolution]
        
        config = types.GenerateContentConfig(**config_params)
        
        response = current_client.models.generate_content(
            model=model_id,
            contents=[image_part, prompt],
            config=config
        )
        
        usage_metadata = response.usage_metadata
        
        return {
            'success': True,
            'response': response.text if response.text else "无响应内容",
            'input_tokens': getattr(usage_metadata, 'prompt_token_count', 0) or 0,
            'output_tokens': getattr(usage_metadata, 'candidates_token_count', 0) or 0,
            'total_tokens': getattr(usage_metadata, 'total_token_count', 0) or 0
        }
        
    except Exception as e:
        logger.error(f"图片分析错误: {str(e)}", exc_info=True)
        return {'success': False, 'error': str(e)}


@app.route('/')
def index():
    """提供前端页面"""
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """处理图片分析请求"""
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 收到新的分析请求")
        
        if 'image' not in request.files:
            return jsonify({'error': '请上传图片文件'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': '请选择图片文件'}), 400
        
        prompt = request.form.get('prompt', '请描述这张图片的内容')
        model_id = request.form.get('model_id', DEFAULT_MODEL_ID)
        if model_id not in AVAILABLE_MODELS:
            model_id = DEFAULT_MODEL_ID
        
        target_width = request.form.get('target_width')
        target_height = request.form.get('target_height')
        
        try:
            target_width = int(target_width) if target_width else None
        except ValueError:
            target_width = None
            
        try:
            target_height = int(target_height) if target_height else None
        except ValueError:
            target_height = None
        
        image_bytes = image_file.read()
        
        file_size_mb = len(image_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return jsonify({'error': f'图片文件过大，请上传小于 {MAX_FILE_SIZE_MB}MB 的图片'}), 400
        
        mime_type = image_file.content_type or 'image/jpeg'
        processed_bytes, image_info = resize_image(image_bytes, target_width, target_height)
        
        format_to_mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'GIF': 'image/gif', 'WEBP': 'image/webp'}
        mime_type = format_to_mime.get(image_info['format'], 'image/jpeg')
        
        media_resolution = request.form.get('media_resolution', None)
        
        result = analyze_image_with_gemini(processed_bytes, prompt, mime_type, model_id, media_resolution)
        
        if result['success']:
            processed_image_base64 = base64.b64encode(processed_bytes).decode('utf-8')
            
            return jsonify({
                'response': result['response'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'total_tokens': result['total_tokens'],
                'image_info': {
                    'original_size': f"{image_info['original_size'][0]}x{image_info['original_size'][1]}",
                    'processed_size': f"{image_info['processed_size'][0]}x{image_info['processed_size'][1]}",
                    'resized': image_info['resized']
                },
                'processed_image_base64': processed_image_base64
            })
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"[{request_id}] 处理请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    return jsonify({
        'models': [{'id': model_id, 'name': name} for model_id, name in AVAILABLE_MODELS.items()],
        'default': DEFAULT_MODEL_ID
    })


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'project_id': PROJECT_ID,
        'default_model_id': DEFAULT_MODEL_ID,
        'available_models': list(AVAILABLE_MODELS.keys()),
        'location': LOCATION,
        'langchain_available': LANGCHAIN_AVAILABLE
    })


def analyze_image_with_langchain(image_bytes: bytes, prompt: str, mime_type: str, model_id: str = None, media_resolution: str = 'medium') -> dict:
    """
    使用 LangChain ChatVertexAI 分析图片
    使用自定义的 ChatVertexAIWithMediaResolution 类，通过 invoke kwargs 传递 media_resolution
    """
    if not LANGCHAIN_AVAILABLE:
        return {
            'success': False,
            'error': 'LangChain not available. Install with: pip install langchain-google-vertexai'
        }
    
    try:
        model_id = model_id or DEFAULT_MODEL_ID
        
        logger.info("=" * 60)
        logger.info(f"[LangChain] 使用 ChatVertexAIWithMediaResolution 分析图片")
        logger.info(f"[LangChain] model_id: {model_id}")
        logger.info(f"[LangChain] media_resolution: {media_resolution}")
        
        # 使用自定义的 ChatVertexAIWithMediaResolution 类
        # media_resolution 通过 default_media_resolution 设置，在 invoke 时自动注入
        logger.info(f"[LangChain] 创建 ChatVertexAIWithMediaResolution 实例...")
        llm = ChatVertexAIWithMediaResolution(
            model=model_id,
            project=PROJECT_ID,
            location=LOCATION,
            default_media_resolution=media_resolution  # 设置默认 media_resolution
        )
        
        logger.info(f"[LangChain] ChatVertexAIWithMediaResolution 实例创建成功")
        logger.info(f"[LangChain] default_media_resolution: {llm.default_media_resolution}")
        
        # 将图片转为 base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 创建包含图片的消息
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )
        
        logger.info(f"[LangChain] 开始调用 ChatVertexAIWithMediaResolution invoke...")
        logger.info(f"[LangChain] media_resolution 将通过 invoke kwargs 传递")
        
        # 调用模型 - media_resolution 会在 invoke 中自动注入
        response = llm.invoke([message])
        
        logger.info(f"[LangChain] 调用成功")
        logger.info(f"[LangChain] response type: {type(response)}")
        
        # 提取 token 使用信息
        usage_metadata = getattr(response, 'usage_metadata', None)
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        if usage_metadata:
            input_tokens = usage_metadata.get('input_tokens', 0)
            output_tokens = usage_metadata.get('output_tokens', 0)
            total_tokens = usage_metadata.get('total_tokens', 0)
            logger.info(f"[LangChain] usage_metadata: {usage_metadata}")
        
        # 提取响应文本
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return {
            'success': True,
            'response': response_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'framework': 'langchain_custom',
            'media_resolution': media_resolution,
            'class_used': 'ChatVertexAIWithMediaResolution'
        }
        
    except Exception as e:
        logger.error(f"[LangChain] 分析错误: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/api/analyze_langchain', methods=['POST'])
def analyze_langchain():
    """使用 LangChain ChatVertexAI 处理图片分析请求 - 默认 media_resolution=medium"""
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info("=" * 60)
        logger.info(f"[{request_id}] 收到 LangChain 分析请求")
        
        if not LANGCHAIN_AVAILABLE:
            return jsonify({'error': 'LangChain not available. Install with: pip install langchain-google-vertexai'}), 500
        
        # 检查是否有图片文件
        if 'image' not in request.files:
            return jsonify({'error': '请上传图片文件'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': '请选择图片文件'}), 400
        
        # 获取参数
        prompt = request.form.get('prompt', '请描述这张图片的内容')
        model_id = request.form.get('model_id', DEFAULT_MODEL_ID)
        media_resolution = request.form.get('media_resolution', 'medium')  # 默认 medium
        
        # 读取图片
        image_bytes = image_file.read()
        mime_type = image_file.content_type or 'image/jpeg'
        
        logger.info(f"[{request_id}] 文件名: {image_file.filename}")
        logger.info(f"[{request_id}] 模型: {model_id}")
        logger.info(f"[{request_id}] Media Resolution: {media_resolution}")
        
        # 调用 LangChain 分析
        result = analyze_image_with_langchain(image_bytes, prompt, mime_type, model_id, media_resolution)
        
        if result['success']:
            return jsonify({
                'response': result['response'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'total_tokens': result['total_tokens'],
                'framework': result['framework'],
                'media_resolution': result['media_resolution'],
                'class_used': result.get('class_used', 'ChatVertexAIWithMediaResolution')
            })
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"[{request_id}] 处理请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500


# ==================== 视频分析功能 ====================

def check_ffmpeg_available() -> bool:
    """检查 ffmpeg 是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

FFMPEG_AVAILABLE = check_ffmpeg_available()
if FFMPEG_AVAILABLE:
    logger.info("ffmpeg 可用，已启用视频音频移除功能")
else:
    logger.warning("ffmpeg 不可用，视频将保留音频轨道")


def remove_audio_from_video(input_path: str, output_path: str) -> bool:
    """
    使用 ffmpeg 移除视频中的音频轨道
    
    Args:
        input_path: 输入视频文件路径
        output_path: 输出视频文件路径
    
    Returns:
        bool: 是否成功移除音频
    """
    if not FFMPEG_AVAILABLE:
        logger.warning("ffmpeg 不可用，跳过音频移除")
        return False
    
    try:
        # 使用 ffmpeg 移除音频: -an 表示不包含音频轨道
        # -c:v copy 表示视频流直接复制，不重新编码
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-an',  # 移除音频
            '-c:v', 'copy',  # 视频流直接复制
            '-y',  # 覆盖输出文件
            output_path
        ]
        
        logger.info(f"[Audio Removal] 执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"[Audio Removal] 成功移除音频")
            return True
        else:
            logger.error(f"[Audio Removal] ffmpeg 错误: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("[Audio Removal] ffmpeg 处理超时")
        return False
    except Exception as e:
        logger.error(f"[Audio Removal] 移除音频时发生错误: {str(e)}")
        return False


def get_video_info(video_path: str) -> dict:
    """获取视频基本信息"""
    if not CV2_AVAILABLE:
        # 返回未知信息，直接模式仍可工作
        return {
            'width': 'unknown',
            'height': 'unknown',
            'fps': 'unknown',
            'frame_count': 'unknown',
            'duration': 'unknown'
        }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': round(fps, 2),
        'frame_count': frame_count,
        'duration': round(duration, 2)
    }


def extract_frames_from_video(video_path: str, fps: float = 1.0, max_frames: int = 30, 
                               target_width: int = None, target_height: int = None) -> list:
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        fps: 提取帧率 (每秒提取多少帧)
        max_frames: 最大帧数
        target_width: 目标宽度
        target_height: 目标高度
    
    Returns:
        list: 帧图片的 bytes 列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)
    frame_interval = max(1, frame_interval)
    
    frames = []
    frame_index = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_interval == 0:
            # 调整帧尺寸
            if target_width or target_height:
                h, w = frame.shape[:2]
                if target_width and target_height:
                    new_w, new_h = target_width, target_height
                elif target_width:
                    new_w = target_width
                    new_h = int(h * (target_width / w))
                else:
                    new_h = target_height
                    new_w = int(w * (target_height / h))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 转换为 JPEG 字节
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(buffer.tobytes())
        
        frame_index += 1
    
    cap.release()
    return frames


def analyze_video_direct_with_gemini(video_bytes: bytes, prompt: str, mime_type: str, 
                                      model_id: str = None, media_resolution: str = None,
                                      video_fps: float = None, start_offset: str = None, 
                                      end_offset: str = None) -> dict:
    """
    直接发送视频给 Gemini 进行分析
    """
    global client
    
    try:
        model_id = model_id or DEFAULT_MODEL_ID
        start_time = time.time()
        
        logger.info(f"[Video Direct] 开始视频分析")
        logger.info(f"[Video Direct] 模型: {model_id}")
        logger.info(f"[Video Direct] MIME: {mime_type}")
        logger.info(f"[Video Direct] media_resolution: {media_resolution}")
        logger.info(f"[Video Direct] video_fps: {video_fps}")
        logger.info(f"[Video Direct] start_offset: {start_offset}, end_offset: {end_offset}")
        
        if model_id in ["gemini-3-pro-preview", "gemini-3-flash-preview"]:
            current_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION_GLOBAL)
        else:
            current_client = client
        
        # 创建视频部分
        video_part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
        
        # 如果有时间裁剪参数，使用 video_metadata
        if start_offset or end_offset:
            video_metadata = {}
            if start_offset:
                # 解析 start_offset (格式: "Xs" 或 "X")
                start_sec = float(start_offset.rstrip('s')) if start_offset else 0
                video_metadata['start_offset'] = f"{start_sec}s"
            if end_offset:
                end_sec = float(end_offset.rstrip('s')) if end_offset else None
                if end_sec:
                    video_metadata['end_offset'] = f"{end_sec}s"
            
            if video_metadata:
                video_part.video_metadata = types.VideoMetadata(**{
                    k: v for k, v in video_metadata.items()
                }) if hasattr(types, 'VideoMetadata') else None
                logger.info(f"[Video Direct] video_metadata: {video_metadata}")
        
        # 配置参数
        config_params = {"max_output_tokens": 4096, "temperature": 0.4}
        
        if model_id == "gemini-2.5-flash-lite":
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        
        if media_resolution and media_resolution in MEDIA_RESOLUTION_MAP:
            config_params["media_resolution"] = MEDIA_RESOLUTION_MAP[media_resolution]
        
        config = types.GenerateContentConfig(**config_params)
        
        response = current_client.models.generate_content(
            model=model_id,
            contents=[video_part, prompt],
            config=config
        )
        
        processing_time = time.time() - start_time
        logger.info(f"[Video Direct] 处理完成，耗时: {processing_time:.2f}秒")
        
        usage_metadata = response.usage_metadata
        
        # 提取详细的 token 信息
        token_details = extract_token_details(usage_metadata)
        
        return {
            'success': True,
            'response': response.text if response.text else "无响应内容",
            'input_tokens': getattr(usage_metadata, 'prompt_token_count', 0) or 0,
            'output_tokens': getattr(usage_metadata, 'candidates_token_count', 0) or 0,
            'total_tokens': getattr(usage_metadata, 'total_token_count', 0) or 0,
            'token_details': token_details,
            'processing_time': round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"[Video Direct] 视频分析错误: {str(e)}", exc_info=True)
        return {'success': False, 'error': str(e)}


def analyze_video_frames_with_gemini(frames: list, prompt: str, model_id: str = None, 
                                      media_resolution: str = None) -> dict:
    """
    发送提取的视频帧给 Gemini 进行分析
    """
    global client
    
    try:
        model_id = model_id or DEFAULT_MODEL_ID
        start_time = time.time()
        
        logger.info(f"[Video Frames] 开始帧分析，帧数: {len(frames)}")
        
        if model_id in ["gemini-3-pro-preview", "gemini-3-flash-preview"]:
            current_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION_GLOBAL)
        else:
            current_client = client
        
        # 创建内容列表
        contents = []
        for i, frame_bytes in enumerate(frames):
            frame_part = types.Part.from_bytes(data=frame_bytes, mime_type='image/jpeg')
            contents.append(frame_part)
        contents.append(f"这是从视频中提取的 {len(frames)} 帧图片。{prompt}")
        
        # 配置参数
        config_params = {"max_output_tokens": 4096, "temperature": 0.4}
        
        if model_id == "gemini-2.5-flash-lite":
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        
        if media_resolution and media_resolution in MEDIA_RESOLUTION_MAP:
            config_params["media_resolution"] = MEDIA_RESOLUTION_MAP[media_resolution]
        
        config = types.GenerateContentConfig(**config_params)
        
        response = current_client.models.generate_content(
            model=model_id,
            contents=contents,
            config=config
        )
        
        processing_time = time.time() - start_time
        logger.info(f"[Video Frames] 处理完成，耗时: {processing_time:.2f}秒")
        
        usage_metadata = response.usage_metadata
        token_details = extract_token_details(usage_metadata)
        
        return {
            'success': True,
            'response': response.text if response.text else "无响应内容",
            'input_tokens': getattr(usage_metadata, 'prompt_token_count', 0) or 0,
            'output_tokens': getattr(usage_metadata, 'candidates_token_count', 0) or 0,
            'total_tokens': getattr(usage_metadata, 'total_token_count', 0) or 0,
            'token_details': token_details,
            'processing_time': round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"[Video Frames] 帧分析错误: {str(e)}", exc_info=True)
        return {'success': False, 'error': str(e)}


def extract_token_details(usage_metadata) -> dict:
    """从 usage_metadata 提取详细的 token 信息"""
    if not usage_metadata:
        return None
    
    details = {
        'prompt_tokens_details': {},
        'candidates_tokens_details': {},
        'cached_content_token_count': 0
    }
    
    # 提取 prompt tokens 详情
    if hasattr(usage_metadata, 'prompt_tokens_details') and usage_metadata.prompt_tokens_details:
        for item in usage_metadata.prompt_tokens_details:
            modality = getattr(item, 'modality', None)
            token_count = getattr(item, 'token_count', 0)
            if modality:
                modality_name = str(modality).split('.')[-1].lower()
                if 'text' in modality_name:
                    details['prompt_tokens_details']['text_tokens'] = token_count
                elif 'image' in modality_name:
                    details['prompt_tokens_details']['image_tokens'] = token_count
                elif 'video' in modality_name:
                    details['prompt_tokens_details']['video_tokens'] = token_count
                elif 'audio' in modality_name:
                    details['prompt_tokens_details']['audio_tokens'] = token_count
    
    # 提取 candidates tokens 详情
    if hasattr(usage_metadata, 'candidates_tokens_details') and usage_metadata.candidates_tokens_details:
        for item in usage_metadata.candidates_tokens_details:
            modality = getattr(item, 'modality', None)
            token_count = getattr(item, 'token_count', 0)
            if modality:
                modality_name = str(modality).split('.')[-1].lower()
                if 'text' in modality_name:
                    details['candidates_tokens_details']['text_tokens'] = token_count
                elif 'image' in modality_name:
                    details['candidates_tokens_details']['image_tokens'] = token_count
                elif 'video' in modality_name:
                    details['candidates_tokens_details']['video_tokens'] = token_count
                elif 'audio' in modality_name:
                    details['candidates_tokens_details']['audio_tokens'] = token_count
    
    # 提取思考 tokens
    if hasattr(usage_metadata, 'thoughts_token_count'):
        details['candidates_tokens_details']['thinking_tokens'] = usage_metadata.thoughts_token_count or 0
    
    # 提取缓存 tokens
    if hasattr(usage_metadata, 'cached_content_token_count'):
        details['cached_content_token_count'] = usage_metadata.cached_content_token_count or 0
    
    return details


@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """处理视频分析请求"""
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 收到视频分析请求")
        
        if 'video' not in request.files:
            return jsonify({'error': '请上传视频文件'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': '请选择视频文件'}), 400
        
        # 获取参数
        prompt = request.form.get('prompt', '请描述这个视频的内容')
        mode = request.form.get('mode', 'direct')  # 'direct' 或 'frames'
        model_id = request.form.get('model_id', DEFAULT_MODEL_ID)
        media_resolution = request.form.get('media_resolution', None)
        
        if model_id not in AVAILABLE_MODELS:
            model_id = DEFAULT_MODEL_ID
        
        # 读取视频
        video_bytes = video_file.read()
        file_size_mb = len(video_bytes) / (1024 * 1024)
        
        logger.info(f"[{request_id}] 文件名: {video_file.filename}")
        logger.info(f"[{request_id}] 文件大小: {file_size_mb:.2f} MB")
        logger.info(f"[{request_id}] 模式: {mode}")
        logger.info(f"[{request_id}] 模型: {model_id}")
        
        if file_size_mb > 100:
            return jsonify({'error': '视频文件过大，请上传小于 100MB 的视频'}), 400
        
        # 获取 MIME 类型
        mime_type = video_file.content_type or 'video/mp4'
        filename = video_file.filename.lower()
        if filename.endswith('.mp4'):
            mime_type = 'video/mp4'
        elif filename.endswith('.webm'):
            mime_type = 'video/webm'
        elif filename.endswith('.mov'):
            mime_type = 'video/quicktime'
        elif filename.endswith('.avi'):
            mime_type = 'video/x-msvideo'
        elif filename.endswith('.ts'):
            mime_type = 'video/mp2t'
        elif filename.endswith('.mkv'):
            mime_type = 'video/x-matroska'
        
        # 保存临时文件以获取视频信息
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            # 获取视频信息
            video_info = get_video_info(tmp_path)
            if not video_info:
                return jsonify({'error': '无法读取视频信息'}), 400
            
            video_info['mode'] = mode
            
            if mode == 'direct':
                # 直接发送视频模式
                video_fps = request.form.get('video_fps')
                start_offset = request.form.get('start_offset')
                end_offset = request.form.get('end_offset')
                
                try:
                    video_fps = float(video_fps) if video_fps else None
                except ValueError:
                    video_fps = None
                
                video_info['video_fps'] = video_fps
                video_info['media_resolution'] = media_resolution
                video_info['start_offset'] = start_offset
                video_info['end_offset'] = end_offset
                
                # 移除视频中的音频轨道
                audio_removed = False
                video_bytes_to_send = video_bytes
                if FFMPEG_AVAILABLE:
                    # 创建输出文件路径
                    no_audio_path = tmp_path + "_noaudio" + os.path.splitext(video_file.filename)[1]
                    if remove_audio_from_video(tmp_path, no_audio_path):
                        # 读取移除音频后的视频
                        with open(no_audio_path, 'rb') as f:
                            video_bytes_to_send = f.read()
                        audio_removed = True
                        logger.info(f"[{request_id}] 已移除音频，原始大小: {len(video_bytes)}, 新大小: {len(video_bytes_to_send)}")
                        # 清理临时文件
                        if os.path.exists(no_audio_path):
                            os.unlink(no_audio_path)
                    else:
                        logger.warning(f"[{request_id}] 音频移除失败，使用原始视频")
                
                video_info['audio_removed'] = audio_removed
                
                result = analyze_video_direct_with_gemini(
                    video_bytes_to_send, prompt, mime_type, model_id, media_resolution,
                    video_fps, start_offset, end_offset
                )
                
                if result['success']:
                    return jsonify({
                        'response': result['response'],
                        'input_tokens': result['input_tokens'],
                        'output_tokens': result['output_tokens'],
                        'total_tokens': result['total_tokens'],
                        'token_details': result.get('token_details'),
                        'video_info': {
                            'width': video_info['width'],
                            'height': video_info['height'],
                            'duration': video_info['duration'],
                            'original_fps': video_info['fps'],
                            'mode': 'direct',
                            'video_fps': video_fps,
                            'media_resolution': media_resolution,
                            'start_offset': start_offset,
                            'end_offset': end_offset,
                            'audio_removed': audio_removed
                        },
                        'processing_time': result.get('processing_time')
                    })
                else:
                    return jsonify({'error': result['error']}), 500
            
            else:
                # 提取帧模式 - 需要 cv2
                if not CV2_AVAILABLE:
                    return jsonify({'error': '帧提取模式需要安装 OpenCV。请使用"直接发送视频"模式，或安装 opencv-python-headless'}), 400
                
                fps = request.form.get('fps', '1')
                max_frames = request.form.get('max_frames', '30')
                frame_width = request.form.get('frame_width')
                frame_height = request.form.get('frame_height')
                
                try:
                    fps = float(fps)
                except ValueError:
                    fps = 1.0
                    
                try:
                    max_frames = int(max_frames)
                except ValueError:
                    max_frames = 30
                    
                try:
                    frame_width = int(frame_width) if frame_width else None
                except ValueError:
                    frame_width = None
                    
                try:
                    frame_height = int(frame_height) if frame_height else None
                except ValueError:
                    frame_height = None
                
                # 提取帧
                frames = extract_frames_from_video(tmp_path, fps, max_frames, frame_width, frame_height)
                
                if not frames:
                    return jsonify({'error': '无法从视频中提取帧'}), 400
                
                video_info['extract_fps'] = fps
                video_info['frames_extracted'] = len(frames)
                if frame_width or frame_height:
                    video_info['frame_size'] = f"{frame_width or 'auto'}x{frame_height or 'auto'}"
                
                result = analyze_video_frames_with_gemini(frames, prompt, model_id, media_resolution)
                
                if result['success']:
                    # 准备帧预览（前6帧）
                    frames_preview = [base64.b64encode(f).decode('utf-8') for f in frames[:6]]
                    
                    return jsonify({
                        'response': result['response'],
                        'input_tokens': result['input_tokens'],
                        'output_tokens': result['output_tokens'],
                        'total_tokens': result['total_tokens'],
                        'token_details': result.get('token_details'),
                        'video_info': {
                            'width': video_info['width'],
                            'height': video_info['height'],
                            'duration': video_info['duration'],
                            'original_fps': video_info['fps'],
                            'mode': 'frames',
                            'extract_fps': fps,
                            'frames_extracted': len(frames),
                            'frame_size': video_info.get('frame_size')
                        },
                        'frames_preview': frames_preview,
                        'processing_time': result.get('processing_time')
                    })
                else:
                    return jsonify({'error': result['error']}), 500
        
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"[{request_id}] 处理视频请求时发生错误: {str(e)}", exc_info=True)
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500


# ==================== InsightFace 人脸识别功能 ====================

# 延迟导入 InsightFace 工具模块
INSIGHTFACE_AVAILABLE = False
try:
    import insightface_utils
    INSIGHTFACE_AVAILABLE = True
    logger.info("InsightFace 模块加载成功")
except ImportError as e:
    logger.warning(f"InsightFace 模块未安装或加载失败: {e}")


@app.route('/api/face/register', methods=['POST'])
def register_face():
    """注册户主人脸"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 收到户主注册请求")
        
        if 'image' not in request.files:
            return jsonify({'error': '请上传照片'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': '请选择照片'}), 400
        
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'error': '请输入户主姓名'}), 400
        
        image_bytes = image_file.read()
        
        logger.info(f"[{request_id}] 注册户主: {name}")
        
        result = insightface_utils.register_face(name, image_bytes)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"[{request_id}] 户主注册失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'注册失败: {str(e)}'}), 500


@app.route('/api/face/list', methods=['GET'])
def list_faces():
    """获取所有已注册的户主列表"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    try:
        faces = insightface_utils.get_all_faces()
        return jsonify({
            'success': True,
            'faces': faces,
            'count': len(faces)
        })
    except Exception as e:
        logger.error(f"获取户主列表失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'获取列表失败: {str(e)}'}), 500


@app.route('/api/face/delete/<name>', methods=['DELETE'])
def delete_face(name):
    """删除户主"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    try:
        logger.info(f"删除户主: {name}")
        result = insightface_utils.delete_face(name)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"删除户主失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'删除失败: {str(e)}'}), 500


@app.route('/api/face/recognize_video', methods=['POST'])
def recognize_video_faces():
    """视频人脸识别"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 收到视频人脸识别请求")
        
        if 'video' not in request.files:
            return jsonify({'error': '请上传视频文件'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': '请选择视频文件'}), 400
        
        # 获取参数
        fps = float(request.form.get('fps', 1.0))
        max_frames = int(request.form.get('max_frames', 30))
        use_gemini = request.form.get('use_gemini', 'false').lower() == 'true'
        prompt = request.form.get('prompt', '请描述视频中人物的行为和场景')
        model_id = request.form.get('model_id', DEFAULT_MODEL_ID)
        
        # 读取视频
        video_bytes = video_file.read()
        file_size_mb = len(video_bytes) / (1024 * 1024)
        
        logger.info(f"[{request_id}] 文件名: {video_file.filename}")
        logger.info(f"[{request_id}] 文件大小: {file_size_mb:.2f} MB")
        logger.info(f"[{request_id}] FPS: {fps}, 最大帧数: {max_frames}")
        logger.info(f"[{request_id}] 使用 Gemini: {use_gemini}")
        
        if file_size_mb > 100:
            return jsonify({'error': '视频文件过大，请上传小于 100MB 的视频'}), 400
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            # 获取视频信息
            video_info = get_video_info(tmp_path)
            
            # InsightFace 人脸识别（带标注图片）
            import time
            start_time = time.time()
            
            # 获取相似度阈值参数（默认 0.7）
            threshold = float(request.form.get('threshold', 0.7))
            
            face_results, annotated_frames = insightface_utils.extract_faces_from_video_with_annotations(
                tmp_path, fps=fps, max_frames=max_frames, threshold=threshold
            )
            
            # 汇总结果
            summary = insightface_utils.summarize_recognition_results(face_results)
            
            face_recognition_time = time.time() - start_time
            logger.info(f"[{request_id}] 人脸识别耗时: {face_recognition_time:.2f}秒，生成 {len(annotated_frames)} 张标注图")
            
            # Gemini 场景分析（可选）
            gemini_analysis = None
            gemini_time = 0
            if use_gemini:
                logger.info(f"[{request_id}] 开始 Gemini 场景分析")
                gemini_start = time.time()
                
                # 获取 MIME 类型
                mime_type = video_file.content_type or 'video/mp4'
                filename = video_file.filename.lower()
                if filename.endswith('.mp4'):
                    mime_type = 'video/mp4'
                elif filename.endswith('.ts'):
                    mime_type = 'video/mp2t'
                elif filename.endswith('.mov'):
                    mime_type = 'video/quicktime'
                
                # 移除音频后分析
                video_bytes_to_send = video_bytes
                if FFMPEG_AVAILABLE:
                    no_audio_path = tmp_path + "_noaudio" + os.path.splitext(video_file.filename)[1]
                    if remove_audio_from_video(tmp_path, no_audio_path):
                        with open(no_audio_path, 'rb') as f:
                            video_bytes_to_send = f.read()
                        if os.path.exists(no_audio_path):
                            os.unlink(no_audio_path)
                
                gemini_result = analyze_video_direct_with_gemini(
                    video_bytes_to_send, prompt, mime_type, model_id
                )
                
                if gemini_result['success']:
                    gemini_analysis = gemini_result['response']
                else:
                    gemini_analysis = f"Gemini 分析失败: {gemini_result.get('error', '未知错误')}"
                
                gemini_time = time.time() - gemini_start
                logger.info(f"[{request_id}] Gemini 分析耗时: {gemini_time:.2f}秒")
            
            total_time = time.time() - start_time
            
            return jsonify({
                'success': True,
                'face_results': face_results,
                'summary': summary,
                'annotated_frames': annotated_frames,
                'gemini_analysis': gemini_analysis,
                'video_info': video_info,
                'processing_time': {
                    'face_recognition': round(face_recognition_time, 2),
                    'gemini_analysis': round(gemini_time, 2),
                    'total': round(total_time, 2)
                }
            })
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"[{request_id}] 视频人脸识别失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'识别失败: {str(e)}'}), 500


@app.route('/api/face/status', methods=['GET'])
def face_status():
    """获取人脸识别模块状态"""
    return jsonify({
        'insightface_available': INSIGHTFACE_AVAILABLE,
        'message': 'InsightFace 模块已加载' if INSIGHTFACE_AVAILABLE else 'InsightFace 模块未安装'
    })


# ==================== 视频标注功能 ====================

# 临时存储标注人物的特征向量（每个会话独立）
import uuid
_label_sessions = {}  # {session_id: {name: embedding}}
_label_output_dir = os.path.join(os.path.dirname(__file__), 'labeled_videos')
os.makedirs(_label_output_dir, exist_ok=True)


@app.route('/api/label/create_session', methods=['POST'])
def create_label_session():
    """创建标注会话"""
    session_id = str(uuid.uuid4())[:8]
    _label_sessions[session_id] = {}
    logger.info(f"创建标注会话: {session_id}")
    return jsonify({
        'success': True,
        'session_id': session_id
    })


@app.route('/api/label/add_person', methods=['POST'])
def add_label_person():
    """添加标注人物"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    try:
        session_id = request.form.get('session_id', '')
        if not session_id or session_id not in _label_sessions:
            return jsonify({'error': '无效的会话ID，请先创建会话'}), 400
        
        if 'image' not in request.files:
            return jsonify({'error': '请上传照片'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': '请选择照片'}), 400
        
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'error': '请输入名字'}), 400
        
        # 名字只允许英文、数字、下划线
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', name):
            return jsonify({'error': '名字只能包含英文字母、数字和下划线'}), 400
        
        image_bytes = image_file.read()
        
        # 提取人脸特征
        embedding = insightface_utils.extract_embedding(image_bytes)
        if embedding is None:
            return jsonify({'error': '未检测到人脸，请上传包含清晰人脸的照片'}), 400
        
        # 保存到会话
        _label_sessions[session_id][name] = embedding
        
        logger.info(f"[会话 {session_id}] 添加人物: {name}")
        
        return jsonify({
            'success': True,
            'message': f'已添加 {name}',
            'name': name,
            'person_count': len(_label_sessions[session_id])
        })
        
    except Exception as e:
        logger.error(f"添加标注人物失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'添加失败: {str(e)}'}), 500


@app.route('/api/label/remove_person', methods=['POST'])
def remove_label_person():
    """移除标注人物"""
    try:
        session_id = request.form.get('session_id', '')
        name = request.form.get('name', '')
        
        if session_id in _label_sessions and name in _label_sessions[session_id]:
            del _label_sessions[session_id][name]
            return jsonify({'success': True, 'message': f'已移除 {name}'})
        
        return jsonify({'error': '人物不存在'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/label/list_persons', methods=['GET'])
def list_label_persons():
    """获取会话中的人物列表"""
    session_id = request.args.get('session_id', '')
    
    if session_id not in _label_sessions:
        return jsonify({'error': '无效的会话ID'}), 400
    
    persons = list(_label_sessions[session_id].keys())
    return jsonify({
        'success': True,
        'persons': persons,
        'count': len(persons)
    })


@app.route('/api/label/process_video', methods=['POST'])
def process_label_video():
    """处理视频标注"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 收到视频标注请求")
        
        session_id = request.form.get('session_id', '')
        if not session_id or session_id not in _label_sessions:
            return jsonify({'error': '无效的会话ID'}), 400
        
        person_embeddings = _label_sessions[session_id]
        if not person_embeddings:
            return jsonify({'error': '请先添加标注人物'}), 400
        
        if 'video' not in request.files:
            return jsonify({'error': '请上传视频文件'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': '请选择视频文件'}), 400
        
        # 获取参数
        threshold = float(request.form.get('threshold', 0.7))
        show_unknown = request.form.get('show_unknown', 'true').lower() == 'true'
        
        # 读取视频
        video_bytes = video_file.read()
        file_size_mb = len(video_bytes) / (1024 * 1024)
        
        logger.info(f"[{request_id}] 文件名: {video_file.filename}, 大小: {file_size_mb:.2f} MB")
        logger.info(f"[{request_id}] 标注人物: {list(person_embeddings.keys())}")
        
        if file_size_mb > 100:
            return jsonify({'error': '视频文件过大，请上传小于 100MB 的视频'}), 400
        
        # 保存临时输入文件
        input_ext = os.path.splitext(video_file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_ext) as tmp:
            tmp.write(video_bytes)
            input_path = tmp.name
        
        # 输出文件路径
        output_filename = f"labeled_{request_id}.mp4"
        output_path = os.path.join(_label_output_dir, output_filename)
        
        try:
            import time
            start_time = time.time()
            
            # 执行视频标注
            result = insightface_utils.label_video_with_faces(
                video_path=input_path,
                person_embeddings=person_embeddings,
                output_path=output_path,
                threshold=threshold,
                show_unknown=show_unknown
            )
            
            processing_time = time.time() - start_time
            
            if result['success']:
                logger.info(f"[{request_id}] 视频标注完成，耗时: {processing_time:.2f}秒")
                
                return jsonify({
                    'success': True,
                    'message': '视频标注完成',
                    'download_url': f'/api/label/download/{output_filename}',
                    'filename': output_filename,
                    'stats': result['stats'],
                    'processing_time': round(processing_time, 2)
                })
            else:
                return jsonify({'error': result['message']}), 500
                
        finally:
            # 清理临时输入文件
            if os.path.exists(input_path):
                os.unlink(input_path)
            
    except Exception as e:
        logger.error(f"[{request_id}] 视频标注失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/api/label/download/<filename>', methods=['GET'])
def download_labeled_video(filename):
    """下载标注后的视频"""
    try:
        # 安全检查：只允许下载 labeled_videos 目录下的文件
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': '无效的文件名'}), 400
        
        file_path = os.path.join(_label_output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 404
        
        return send_from_directory(_label_output_dir, filename, as_attachment=True)
        
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== 实时标注功能 (SSE) ====================

# 实时标注会话存储
_realtime_sessions = {}  # {session_id: {name: embedding}}


@app.route('/api/realtime/create_session', methods=['POST'])
def create_realtime_session():
    """创建实时标注会话"""
    session_id = str(uuid.uuid4())[:8]
    _realtime_sessions[session_id] = {}
    logger.info(f"创建实时标注会话: {session_id}")
    return jsonify({
        'success': True,
        'session_id': session_id
    })


@app.route('/api/realtime/add_person', methods=['POST'])
def add_realtime_person():
    """添加实时标注人物"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    try:
        session_id = request.form.get('session_id', '')
        if not session_id or session_id not in _realtime_sessions:
            return jsonify({'error': '无效的会话ID，请先创建会话'}), 400
        
        if 'image' not in request.files:
            return jsonify({'error': '请上传照片'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': '请选择照片'}), 400
        
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'error': '请输入名字'}), 400
        
        # 名字只允许英文、数字、下划线
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', name):
            return jsonify({'error': '名字只能包含英文字母、数字和下划线'}), 400
        
        image_bytes = image_file.read()
        
        # 提取人脸特征
        embedding = insightface_utils.extract_embedding(image_bytes)
        if embedding is None:
            return jsonify({'error': '未检测到人脸，请上传包含清晰人脸的照片'}), 400
        
        # 保存到会话
        _realtime_sessions[session_id][name] = embedding
        
        logger.info(f"[实时会话 {session_id}] 添加人物: {name}")
        
        return jsonify({
            'success': True,
            'message': f'已添加 {name}',
            'name': name,
            'person_count': len(_realtime_sessions[session_id])
        })
        
    except Exception as e:
        logger.error(f"添加实时标注人物失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'添加失败: {str(e)}'}), 500


@app.route('/api/realtime/remove_person', methods=['POST'])
def remove_realtime_person():
    """移除实时标注人物"""
    try:
        session_id = request.form.get('session_id', '')
        name = request.form.get('name', '')
        
        if session_id in _realtime_sessions and name in _realtime_sessions[session_id]:
            del _realtime_sessions[session_id][name]
            return jsonify({'success': True, 'message': f'已移除 {name}'})
        
        return jsonify({'error': '人物不存在'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/realtime/stream', methods=['POST'])
def realtime_stream():
    """实时标注 SSE 流"""
    if not INSIGHTFACE_AVAILABLE:
        return jsonify({'error': 'InsightFace 模块未安装'}), 500
    
    try:
        session_id = request.form.get('session_id', '')
        if not session_id or session_id not in _realtime_sessions:
            return jsonify({'error': '无效的会话ID'}), 400
        
        person_embeddings = _realtime_sessions[session_id]
        if not person_embeddings:
            return jsonify({'error': '请先添加标注人物'}), 400
        
        if 'video' not in request.files:
            return jsonify({'error': '请上传视频文件'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': '请选择视频文件'}), 400
        
        # 获取参数
        threshold = float(request.form.get('threshold', 0.7))
        show_unknown = request.form.get('show_unknown', 'true').lower() == 'true'
        target_fps = float(request.form.get('target_fps', 2.0))
        enable_gemini = request.form.get('enable_gemini', 'false').lower() == 'true'
        gemini_prompt = request.form.get('gemini_prompt', '').strip()
        gemini_mode = request.form.get('gemini_mode', 'frames')  # 'frames' 或 'direct'
        
        # 读取视频到临时文件
        video_bytes = video_file.read()
        input_ext = os.path.splitext(video_file.filename)[1] or '.mp4'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_ext) as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name
        
        logger.info(f"实时标注开始: session={session_id}, 人物={list(person_embeddings.keys())}, enable_gemini={enable_gemini}, gemini_mode={gemini_mode}")
        
        def sample_key_frames(appearances, max_frames=6):
            """
            从出场帧中采样关键帧
            策略：首帧 + 末帧 + 中间等距采样
            """
            if not appearances:
                return []
            if len(appearances) <= max_frames:
                return appearances
            
            # 首帧必选
            sampled = [appearances[0]]
            
            # 末帧必选
            if len(appearances) > 1:
                sampled.append(appearances[-1])
            
            # 中间等距采样
            remaining_slots = max_frames - len(sampled)
            if remaining_slots > 0 and len(appearances) > 2:
                middle_frames = appearances[1:-1]
                step = len(middle_frames) / (remaining_slots + 1)
                for i in range(remaining_slots):
                    idx = int((i + 1) * step)
                    if idx < len(middle_frames):
                        sampled.insert(-1, middle_frames[idx])
            
            # 按时间排序
            sampled.sort(key=lambda x: x['time_seconds'])
            return sampled[:max_frames]
        
        def generate():
            gemini_analysis = None
            owner_appearances = None
            try:
                for result in insightface_utils.realtime_label_generator(
                    video_path=video_path,
                    person_embeddings=person_embeddings,
                    threshold=threshold,
                    show_unknown=show_unknown,
                    target_fps=target_fps,
                    collect_appearances=enable_gemini  # 启用 Gemini 时收集出场帧
                ):
                    # 如果是完成信号且启用了Gemini分析
                    if result.get('type') == 'complete' and enable_gemini:
                        # 获取 owner 出场帧数据
                        owner_appearances = result.pop('owner_appearances', {})
                        
                        # 先发送完成信号（不含Gemini结果）
                        yield f"data: {json.dumps(result)}\n\n"
                        
                        # 发送Gemini分析中的提示
                        yield f"data: {json.dumps({'type': 'gemini_analyzing'})}\n\n"
                        
                        logger.info(f"开始 Gemini 场景分析... 模式={gemini_mode}, 收集到 {len(owner_appearances)} 个人物的出场帧")
                        
                        try:
                            token_stats = {}
                            
                            # 根据模式选择不同的分析方式
                            if gemini_mode == 'direct':
                                # 直接发送视频模式（结合 InsightFace 识别结果）
                                logger.info("使用直接发送视频模式分析（结合 InsightFace 识别结果）...")
                                
                                # 构建 Owner 出场信息（基于 InsightFace 识别结果）
                                owner_info_lines = []
                                if owner_appearances:
                                    for name, appearances in owner_appearances.items():
                                        time_points = [app['time'] for app in appearances]
                                        owner_info_lines.append(f"  - {name} (Owner): 出现于 {', '.join(time_points)}")
                                
                                owner_info_text = '\n'.join(owner_info_lines) if owner_info_lines else "  （未检测到已标注人物）"
                                
                                # 用户提示词
                                user_prompt = gemini_prompt if gemini_prompt else "请分析视频中的场景内容。"
                                
                                # 构建提示词（结合 InsightFace 识别结果）
                                prompt = f"""{user_prompt}

【InsightFace 人脸识别结果】
以下是通过人脸识别技术确认的已标注人物 (Owner) 及其出现时间：
{owner_info_text}

【标注说明】
- Owner: 已通过照片标注的人物，请在描述中使用其名字
- 未标注的人物请称为"陌生人"或"其他人"
- 请结合识别结果描述视频内容"""

                                # 确定 mime_type
                                mime_type_map = {
                                    '.mp4': 'video/mp4',
                                    '.avi': 'video/x-msvideo',
                                    '.mov': 'video/quicktime',
                                    '.mkv': 'video/x-matroska',
                                    '.webm': 'video/webm'
                                }
                                video_mime_type = mime_type_map.get(input_ext.lower(), 'video/mp4')
                                
                                # 创建视频部分
                                video_part = types.Part.from_bytes(data=video_bytes, mime_type=video_mime_type)
                                
                                config = types.GenerateContentConfig(
                                    max_output_tokens=4096,
                                    temperature=0.4
                                )
                                
                                response = client.models.generate_content(
                                    model=DEFAULT_MODEL_ID,
                                    contents=[video_part, prompt],
                                    config=config
                                )
                                
                                if response and response.text:
                                    gemini_analysis = response.text.strip()
                                else:
                                    gemini_analysis = "分析未返回结果"
                                
                                # 收集 Token 信息
                                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                    um = response.usage_metadata
                                    token_stats = {
                                        'input_tokens': getattr(um, 'prompt_token_count', 0),
                                        'output_tokens': getattr(um, 'candidates_token_count', 0),
                                        'total_tokens': getattr(um, 'total_token_count', 0),
                                        'persons_analyzed': len(owner_appearances),
                                        'mode': 'direct'
                                    }
                                
                                logger.info(f"直接视频分析完成, Token: {token_stats}")
                                
                            else:
                                # 提取帧模式（原有逻辑）
                                person_summaries = {}
                            
                                # 为每个 owner 生成场景描述
                                if owner_appearances:
                                    for person_name, appearances in owner_appearances.items():
                                        # 采样关键帧
                                        key_frames = sample_key_frames(appearances, max_frames=6)
                                    
                                    if not key_frames:
                                        continue
                                    
                                    logger.info(f"分析 {person_name} 的 {len(key_frames)} 个关键帧...")
                                    
                                    # 构建 Gemini Prompt（结合用户自定义提示词）
                                    time_points = [f['time'] for f in key_frames]
                                    
                                    # 使用用户自定义提示词或默认提示词
                                    user_prompt = gemini_prompt if gemini_prompt else "请分析每个标注人物在视频中的场景，描述他们的动作、姿态和周围环境。"
                                    
                                    prompt = f"""以下是视频中通过人脸识别确认的人物 "{person_name}" 在不同时间点的画面截图。

【用户分析需求】
{user_prompt}

【人物信息】
- 人物名称: {person_name}
- 出现时间点: {', '.join(time_points)}

【输出格式要求】
请按以下格式输出（每个时间点一行）:
[时间] - [场景描述]

注意：
1. 描述要简洁明了，每条不超过50字
2. 重点关注用户分析需求中提到的内容
3. 如果多个时间点场景相似，可以合并描述"""

                                    # 构建多图片请求内容
                                    contents = []
                                    for frame in key_frames:
                                        # 将 base64 解码为 bytes，使用项目中已有的 types.Part
                                        frame_bytes = base64.b64decode(frame['image_base64'])
                                        frame_part = types.Part.from_bytes(data=frame_bytes, mime_type='image/jpeg')
                                        contents.append(frame_part)
                                    # 添加文本 prompt
                                    contents.append(prompt)
                                    
                                    # 调用 Gemini 分析（使用项目中已有的 client）
                                    try:
                                        config = types.GenerateContentConfig(
                                            max_output_tokens=2048,
                                            temperature=0.4
                                        )
                                        response = client.models.generate_content(
                                            model=DEFAULT_MODEL_ID,
                                            contents=contents,
                                            config=config
                                        )
                                        
                                        if response and response.text:
                                            # 收集 Token 信息
                                            person_token_info = {}
                                            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                                um = response.usage_metadata
                                                person_token_info = {
                                                    'input_tokens': getattr(um, 'prompt_token_count', 0),
                                                    'output_tokens': getattr(um, 'candidates_token_count', 0),
                                                    'total_tokens': getattr(um, 'total_token_count', 0)
                                                }
                                            
                                            person_summaries[person_name] = {
                                                'time_points': time_points,
                                                'analysis': response.text.strip(),
                                                'token_info': person_token_info
                                            }
                                        else:
                                            person_summaries[person_name] = {
                                                'time_points': time_points,
                                                'analysis': '分析未返回结果',
                                                'token_info': {}
                                            }
                                    except Exception as e:
                                        logger.error(f"分析 {person_name} 失败: {e}")
                                        person_summaries[person_name] = {
                                            'time_points': time_points,
                                            'analysis': f'分析失败: {str(e)}',
                                            'token_info': {}
                                        }
                                
                                # 构建最终分析结果和汇总 Token 信息（提取帧模式）
                                total_input_tokens = 0
                                total_output_tokens = 0
                                total_tokens = 0
                                
                                if person_summaries:
                                    gemini_analysis = "【人物场景分析】\n\n"
                                    for person_name, summary in person_summaries.items():
                                        gemini_analysis += f"━━━ {person_name} ━━━\n"
                                        gemini_analysis += f"出现时间点: {', '.join(summary['time_points'])}\n\n"
                                        gemini_analysis += f"{summary['analysis']}\n\n"
                                        
                                        # 汇总 Token
                                        if summary.get('token_info'):
                                            total_input_tokens += summary['token_info'].get('input_tokens', 0)
                                            total_output_tokens += summary['token_info'].get('output_tokens', 0)
                                            total_tokens += summary['token_info'].get('total_tokens', 0)
                                else:
                                    gemini_analysis = "未检测到标注人物的出场帧，无法生成场景分析"
                                
                                # Token 统计信息
                                token_stats = {
                                    'input_tokens': total_input_tokens,
                                    'output_tokens': total_output_tokens,
                                    'total_tokens': total_tokens,
                                    'persons_analyzed': len(person_summaries),
                                    'frames_analyzed': sum(len(s.get('time_points', [])) for s in person_summaries.values()),
                                    'mode': 'frames'
                                }
                                
                                logger.info(f"Gemini 人物场景分析完成, Token: {token_stats}")
                            
                        except Exception as e:
                            logger.error(f"Gemini 分析失败: {e}")
                            gemini_analysis = f"分析失败: {str(e)}"
                            token_stats = {}
                        
                        # 发送Gemini分析结果（包含 Token 统计）
                        yield f"data: {json.dumps({'type': 'gemini_complete', 'gemini_analysis': gemini_analysis, 'token_stats': token_stats})}\n\n"
                    else:
                        yield f"data: {json.dumps(result)}\n\n"
            finally:
                # 清理临时文件
                if os.path.exists(video_path):
                    os.unlink(video_path)
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"实时标注失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


# ==================== GCloud 认证配置 API ====================

@app.route('/api/gcloud/set_project', methods=['POST'])
def gcloud_set_project():
    """设置 GCloud 项目"""
    import subprocess
    import shlex
    
    project_id = request.form.get('project_id', '').strip()
    if not project_id:
        return jsonify({'error': '请输入 Project ID'}), 400
    
    # 验证 project_id 格式（只允许字母、数字、连字符）
    if not all(c.isalnum() or c == '-' for c in project_id):
        return jsonify({'error': 'Project ID 格式无效'}), 400
    
    def generate():
        cmd = f"gcloud config set project {shlex.quote(project_id)}"
        yield f"data: {json.dumps({'type': 'start', 'command': cmd})}\n\n"
        
        try:
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'type': 'output', 'line': line.strip()})}\n\n"
            
            process.wait()
            success = process.returncode == 0
            message = '项目设置成功' if success else '项目设置失败'
            yield f"data: {json.dumps({'type': 'complete', 'success': success, 'message': message})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


_gcloud_auth_process = None

@app.route('/api/gcloud/auth', methods=['POST'])
def gcloud_auth():
    """执行 GCloud 认证 - 第一步：获取认证URL"""
    global _gcloud_auth_process
    import subprocess
    
    # 如果有旧进程，先清理
    if _gcloud_auth_process is not None:
        try:
            _gcloud_auth_process.kill()
        except:
            pass
        _gcloud_auth_process = None
    
    def generate():
        global _gcloud_auth_process
        cmd = "gcloud auth application-default login --no-launch-browser --quiet"
        yield f"data: {json.dumps({'type': 'start', 'command': cmd})}\n\n"
        
        try:
            _gcloud_auth_process = subprocess.Popen(
                cmd, shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            
            # 读取输出直到看到 URL（表示需要用户去浏览器认证）
            url_found = False
            while True:
                line = _gcloud_auth_process.stdout.readline()
                if not line:
                    # 没有更多输出，检查进程是否还在运行
                    if _gcloud_auth_process.poll() is None and url_found:
                        # 进程还在运行且已发现URL，说明在等待输入
                        yield f"data: {json.dumps({'type': 'waiting_code', 'message': '请在下方输入验证码'})}\n\n"
                        return
                    break
                    
                line_stripped = line.strip()
                if line_stripped:  # 只输出非空行
                    yield f"data: {json.dumps({'type': 'output', 'line': line_stripped})}\n\n"
                
                # 检测 URL - 发现后立即显示验证码输入框
                if 'https://accounts.google.com' in line_stripped or (line_stripped.startswith('http') and 'oauth' in line_stripped.lower()):
                    url_found = True
                    yield f"data: {json.dumps({'type': 'waiting_code', 'message': '请在下方输入验证码'})}\n\n"
                    return  # 保持进程运行，等待验证码
            
            # 如果进程已结束
            if _gcloud_auth_process.poll() is not None:
                success = _gcloud_auth_process.returncode == 0
                message = '认证成功' if success else '认证失败或已取消'
                yield f"data: {json.dumps({'type': 'complete', 'success': success, 'message': message})}\n\n"
                _gcloud_auth_process = None
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            _gcloud_auth_process = None
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/gcloud/auth_code', methods=['POST'])
def gcloud_auth_code():
    """执行 GCloud 认证 - 第二步：提交验证码"""
    global _gcloud_auth_process
    import subprocess
    
    auth_code = request.form.get('auth_code', '').strip()
    if not auth_code:
        return jsonify({'error': '请输入验证码'}), 400
    
    if _gcloud_auth_process is None:
        return jsonify({'error': '认证进程不存在，请重新执行认证'}), 400
    
    def generate():
        global _gcloud_auth_process
        
        try:
            # 发送验证码
            _gcloud_auth_process.stdin.write(auth_code + '\n')
            _gcloud_auth_process.stdin.flush()
            
            yield f"data: {json.dumps({'type': 'output', 'line': '正在验证...'})}\n\n"
            
            # 读取剩余输出
            while True:
                line = _gcloud_auth_process.stdout.readline()
                if not line:
                    break
                yield f"data: {json.dumps({'type': 'output', 'line': line.strip()})}\n\n"
            
            _gcloud_auth_process.wait()
            success = _gcloud_auth_process.returncode == 0
            message = '认证成功' if success else '认证失败'
            yield f"data: {json.dumps({'type': 'complete', 'success': success, 'message': message})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            _gcloud_auth_process = None
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/gcloud/status', methods=['GET'])
def gcloud_status():
    """检查 GCloud 认证状态"""
    import subprocess
    
    try:
        # 检查当前项目
        project_result = subprocess.run(
            "gcloud config get-value project",
            shell=True, capture_output=True, text=True, timeout=10
        )
        current_project = project_result.stdout.strip() if project_result.returncode == 0 else ''
        
        # 检查认证状态
        auth_result = subprocess.run(
            "gcloud auth application-default print-access-token",
            shell=True, capture_output=True, text=True, timeout=10
        )
        is_authenticated = auth_result.returncode == 0
        
        # 检查账号
        account_result = subprocess.run(
            "gcloud config get-value account",
            shell=True, capture_output=True, text=True, timeout=10
        )
        current_account = account_result.stdout.strip() if account_result.returncode == 0 else ''
        
        return jsonify({
            'success': True,
            'project': current_project,
            'account': current_account,
            'authenticated': is_authenticated
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': '检查超时'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== 陌生人检测 API ====================

# 存储陌生人检测 Cache 信息
stranger_caches = {}


def compute_cache_fingerprint(owner_names: list, base_prompt: str = '') -> str:
    """
    计算 Cache 指纹，用于判断是否可以复用已有 Cache
    基于户主名称列表和基础提示词生成唯一标识
    """
    import hashlib
    # 对户主名称排序后拼接
    sorted_names = sorted(owner_names)
    content = '|'.join(sorted_names) + '||' + base_prompt.strip()
    return hashlib.md5(content.encode()).hexdigest()[:16]


def find_reusable_cache(fingerprint: str) -> dict:
    """
    查找可复用的 Cache
    返回: {'found': bool, 'cache_id': str, 'cache_info': dict, 'remaining_seconds': int}
    """
    current_time = time.time()
    
    for cache_id, cache_info in list(stranger_caches.items()):
        # 检查指纹是否匹配
        if cache_info.get('fingerprint') != fingerprint:
            continue
        
        # 检查是否过期
        expires_timestamp = cache_info.get('expires_timestamp', 0)
        if current_time >= expires_timestamp:
            # 已过期，从本地记录中删除（但不删除远端，让它自动过期）
            logger.info(f"Cache {cache_id} 已过期，移除本地记录")
            del stranger_caches[cache_id]
            continue
        
        # 找到可复用的 Cache
        remaining_seconds = int(expires_timestamp - current_time)
        logger.info(f"找到可复用的 Cache: {cache_id}, 剩余 {remaining_seconds} 秒")
        return {
            'found': True,
            'cache_id': cache_id,
            'cache_info': cache_info,
            'remaining_seconds': remaining_seconds
        }
    
    return {'found': False, 'cache_id': None, 'cache_info': None, 'remaining_seconds': 0}


def cleanup_expired_caches():
    """清理已过期的 Cache 记录"""
    current_time = time.time()
    expired_ids = []
    
    for cache_id, cache_info in stranger_caches.items():
        expires_timestamp = cache_info.get('expires_timestamp', 0)
        if current_time >= expires_timestamp:
            expired_ids.append(cache_id)
    
    for cache_id in expired_ids:
        logger.info(f"清理过期 Cache 记录: {cache_id}")
        del stranger_caches[cache_id]
    
    return len(expired_ids)


def detect_stranger_in_analysis(analysis_text: str) -> bool:
    """
    智能判断分析结果中是否检测到陌生人
    使用更精确的模式匹配，避免误判
    
    Returns:
        bool: True 表示检测到陌生人，False 表示未检测到
    """
    import re
    
    text = analysis_text.lower()
    
    # 1. 检测明确的否定表达（优先级最高）
    # 匹配 "是否发现陌生人：否" 或类似表达
    if re.search(r'是否发现陌生人[：:]\s*否', analysis_text):
        return False
    
    # 匹配 "陌生人数量：0" 或 "陌生人数量: 0"
    if re.search(r'陌生人数量[：:]\s*0', analysis_text):
        return False
    
    # 匹配 "没有发现陌生人" "未发现陌生人" "无陌生人"
    if re.search(r'(没有|未|无|不存在).{0,5}陌生人', analysis_text):
        return False
    
    # 匹配英文表达
    if re.search(r'no\s+stranger', text):
        return False
    
    # 2. 检测明确的肯定表达
    # 匹配 "是否发现陌生人：是"
    if re.search(r'是否发现陌生人[：:]\s*是', analysis_text):
        return True
    
    # 匹配 "陌生人数量：[非0数字]"
    match = re.search(r'陌生人数量[：:]\s*(\d+)', analysis_text)
    if match:
        count = int(match.group(1))
        if count > 0:
            return True
        else:
            return False
    
    # 匹配 "发现陌生人" "检测到陌生人" "出现陌生人"（但不是否定句）
    if re.search(r'(发现|检测到|出现|存在).{0,5}陌生人', analysis_text):
        # 确保不是否定句
        if not re.search(r'(没有|未|无|不存在).{0,10}(发现|检测到|出现|存在).{0,5}陌生人', analysis_text):
            return True
    
    # 3. 默认返回 False（未检测到陌生人）
    return False


@app.route('/api/stranger/analyze', methods=['POST'])
def stranger_analyze():
    """直接分析模式：发送视频+户主照片给 Gemini 判断陌生人"""
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 收到陌生人检测请求（直接分析模式）")
        
        # 获取视频
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '请上传视频文件'}), 400
        
        video_file = request.files['video']
        if not video_file.filename:
            return jsonify({'success': False, 'error': '视频文件名为空'}), 400
        
        # 获取户主数据
        owners_json = request.form.get('owners', '[]')
        try:
            owners = json.loads(owners_json)
        except:
            owners = []
        
        if not owners:
            return jsonify({'success': False, 'error': '请至少添加一个户主照片'}), 400
        
        # 获取提示词
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            prompt = "请分析视频中是否有陌生人出现，并描述场景。"
        
        # 获取媒体分辨率参数
        media_resolution = request.form.get('media_resolution', None)
        logger.info(f"[{request_id}] 媒体分辨率: {media_resolution}")
        
        # 读取视频
        video_bytes = video_file.read()
        video_size_mb = len(video_bytes) / 1024 / 1024
        logger.info(f"[{request_id}] 视频大小: {video_size_mb:.2f} MB, 户主数量: {len(owners)}")
        
        # 确定视频 mime_type
        filename = video_file.filename.lower()
        mime_type_map = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.ts': 'video/mp2t'
        }
        ext = os.path.splitext(filename)[1]
        video_mime_type = mime_type_map.get(ext, 'video/mp4')
        
        # 构建 contents
        contents = []
        
        # 添加户主照片
        owner_info_text = "【已知户主照片】\n请记住以下户主的面部特征：\n"
        for owner in owners:
            name = owner.get('name', '未知')
            image_data = owner.get('imageData', '')
            
            if image_data:
                # 解码 base64 图片
                if image_data.startswith('data:'):
                    image_data = image_data.split(',')[1]
                
                try:
                    img_bytes = base64.b64decode(image_data)
                    img_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
                    contents.append(img_part)
                    owner_info_text += f"- {name}\n"
                except Exception as e:
                    logger.warning(f"解码户主 {name} 照片失败: {e}")
        
        contents.append(owner_info_text)
        
        # 添加视频
        video_part = types.Part.from_bytes(data=video_bytes, mime_type=video_mime_type)
        contents.append(video_part)
        
        # 添加分析指令
        analysis_instruction = f"""
【分析任务】
{prompt}

【识别规则】
1. 对比视频中出现的人脸与上述户主照片
2. 如果匹配，标记为"户主"并说明是哪位户主
3. 如果不匹配任何户主照片，标记为"⚠️陌生人"
4. 描述每个人的行为和所在场景

【输出要求】
请按以下格式输出分析结果：

🔍 **检测结果**
- 是否发现陌生人：是/否
- 识别到的户主：[列出识别到的户主名称]
- 陌生人数量：[数量]

📝 **详细分析**
[详细描述每个人物的出现时间、行为和场景]

📊 **场景总结**
[整体场景描述]
"""
        contents.append(analysis_instruction)
        
        # 调用 Gemini
        start_time = time.time()
        
        # 配置参数（包含媒体分辨率）
        config_params = {
            'max_output_tokens': 4096,
            'temperature': 0.4
        }
        
        # 应用媒体分辨率设置
        if media_resolution and media_resolution in MEDIA_RESOLUTION_MAP:
            config_params["media_resolution"] = MEDIA_RESOLUTION_MAP[media_resolution]
            logger.info(f"[{request_id}] 应用媒体分辨率: {media_resolution}")
        
        config = types.GenerateContentConfig(**config_params)
        
        response = client.models.generate_content(
            model=DEFAULT_MODEL_ID,
            contents=contents,
            config=config
        )
        
        processing_time = time.time() - start_time
        
        # 提取结果
        analysis_text = response.text if response.text else "分析未返回结果"
        
        # 使用智能判断函数判断是否有陌生人
        has_stranger = detect_stranger_in_analysis(analysis_text)
        logger.info(f"[{request_id}] 陌生人检测结果: has_stranger={has_stranger}")
        
        # Token 统计（包含详细信息）
        token_stats = {}
        token_details = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            token_stats = {
                'input_tokens': getattr(um, 'prompt_token_count', 0),
                'output_tokens': getattr(um, 'candidates_token_count', 0),
                'total_tokens': getattr(um, 'total_token_count', 0)
            }
            # 提取详细 token 信息
            token_details = extract_token_details(um)
        
        logger.info(f"[{request_id}] 分析完成，耗时: {processing_time:.2f}秒, Token: {token_stats}")
        
        return jsonify({
            'success': True,
            'has_stranger': has_stranger,
            'analysis': analysis_text,
            'token_stats': token_stats,
            'token_details': token_details,
            'processing_time': round(processing_time, 2),
            'owners_count': len(owners)
        })
        
    except Exception as e:
        logger.error(f"[{request_id}] 陌生人检测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stranger/create_cache', methods=['POST'])
def stranger_create_cache():
    """创建陌生人检测 Cache（只 Cache 系统指令和户主照片）- 支持智能复用"""
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 创建陌生人检测 Cache（智能复用模式）")
        
        # 先清理过期的 Cache
        cleanup_expired_caches()
        
        # 获取户主数据
        owners_json = request.form.get('owners', '[]')
        try:
            owners = json.loads(owners_json)
        except:
            owners = []
        
        if not owners:
            return jsonify({'success': False, 'error': '请至少添加一个户主照片'}), 400
        
        # 获取自定义提示词
        base_prompt = request.form.get('prompt', '').strip()
        
        # 是否强制刷新 Cache
        force_refresh = request.form.get('force_refresh', 'false').lower() == 'true'
        
        # 提取户主名称列表
        owner_names_for_fingerprint = [owner.get('name', '未知') for owner in owners]
        
        # 计算 Cache 指纹
        fingerprint = compute_cache_fingerprint(owner_names_for_fingerprint, base_prompt)
        logger.info(f"[{request_id}] Cache 指纹: {fingerprint}, 户主: {owner_names_for_fingerprint}")
        
        # 查找可复用的 Cache（非强制刷新模式）
        if not force_refresh:
            reusable = find_reusable_cache(fingerprint)
            if reusable['found']:
                cache_id = reusable['cache_id']
                cache_info = reusable['cache_info']
                remaining_seconds = reusable['remaining_seconds']
                
                logger.info(f"[{request_id}] 复用已有 Cache: {cache_id}, 剩余 {remaining_seconds} 秒")
                
                return jsonify({
                    'success': True,
                    'cache_id': cache_id,
                    'owner_names': cache_info['owner_names'],
                    'owner_count': cache_info['owner_count'],
                    'cache_tokens': cache_info.get('cache_tokens', 0),
                    'created_at': cache_info['created_at'],
                    'expires_at': cache_info['expires_at'],
                    'reused': True,  # 标记为复用
                    'remaining_seconds': remaining_seconds
                })
        
        # 构建系统指令
        system_instruction = """你是一个安全监控AI助手，专门负责分析视频中的人物，判断是否有陌生人出现。

【你的任务】
1. 分析用户上传的视频
2. 对比视频中出现的人脸与已知户主照片
3. 判断是否有陌生人（未在户主照片中出现的人）
4. 描述场景和人物行为

【识别规则】
- 如果视频中的人与户主照片匹配，标记为"✅ 户主 - [姓名]"
- 如果视频中的人与任何户主都不匹配，标记为"⚠️ 陌生人"
- 对于每个人，描述其行为、位置和出现时间

【输出格式】
每次分析请按以下格式输出：

🔍 **检测结果**
- 是否发现陌生人：是/否
- 识别到的户主：[列出]
- 陌生人数量：[数量]

📝 **详细分析**
[每个人的详细描述]

📊 **场景总结**
[整体场景描述]
"""
        
        if base_prompt:
            system_instruction += f"\n【用户特别要求】\n{base_prompt}\n"
        
        # 构建 Cache 内容（包含户主照片）
        cache_contents = []
        owner_names = []
        
        system_instruction += "\n【已知户主照片】\n以下是已注册户主的照片，请记住他们的面部特征：\n"
        
        for owner in owners:
            name = owner.get('name', '未知')
            image_data = owner.get('imageData', '')
            owner_names.append(name)
            
            if image_data:
                if image_data.startswith('data:'):
                    image_data = image_data.split(',')[1]
                
                try:
                    img_bytes = base64.b64decode(image_data)
                    img_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
                    cache_contents.append(img_part)
                    system_instruction += f"- {name} (照片如上)\n"
                except Exception as e:
                    logger.warning(f"解码户主 {name} 照片失败: {e}")
                    system_instruction += f"- {name} (照片解码失败)\n"
        
        # 添加说明文本
        cache_contents.append(f"以上是 {len(owner_names)} 位户主的照片，分别是: {', '.join(owner_names)}")
        
        # 创建 Cache
        cache_model = "gemini-2.0-flash-001"
        
        try:
            cache = client.caches.create(
                model=cache_model,
                config=types.CreateCachedContentConfig(
                    display_name=f"stranger-detect-{request_id}",
                    system_instruction=system_instruction,
                    contents=cache_contents,
                    ttl="3600s"  # 1小时有效期
                )
            )
            
            cache_id = cache.name.split('/')[-1] if '/' in cache.name else cache.name
            
            # 获取 Token 信息
            cache_tokens = 0
            if hasattr(cache, 'usage_metadata') and cache.usage_metadata:
                cache_tokens = getattr(cache.usage_metadata, 'total_token_count', 0)
            
            # 存储 Cache 信息（包含指纹和过期时间戳，用于复用匹配）
            current_time = time.time()
            expires_timestamp = current_time + 3600  # 1小时后过期
            
            stranger_caches[cache_id] = {
                'cache_name': cache.name,
                'model': cache_model,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
                'expires_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expires_timestamp)),
                'expires_timestamp': expires_timestamp,  # 用于过期检查
                'fingerprint': fingerprint,  # 用于复用匹配
                'owner_names': owner_names,
                'owner_count': len(owners),
                'cache_tokens': cache_tokens
            }
            
            logger.info(f"[{request_id}] 新 Cache 创建成功: {cache_id}, Token: {cache_tokens}, 指纹: {fingerprint}")
            
            return jsonify({
                'success': True,
                'cache_id': cache_id,
                'owner_names': owner_names,
                'owner_count': len(owners),
                'cache_tokens': cache_tokens,
                'created_at': stranger_caches[cache_id]['created_at'],
                'expires_at': stranger_caches[cache_id]['expires_at'],
                'reused': False,  # 标记为新建
                'remaining_seconds': 3600
            })
            
        except Exception as cache_error:
            logger.error(f"Cache 创建失败: {cache_error}")
            return jsonify({
                'success': False,
                'error': f'Cache 创建失败: {str(cache_error)}'
            }), 500
        
    except Exception as e:
        logger.error(f"[{request_id}] 创建 Cache 失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stranger/query_with_cache', methods=['POST'])
def stranger_query_with_cache():
    """使用 Cache 分析视频（视频不 Cache，每次发送）"""
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    try:
        logger.info(f"[{request_id}] 收到陌生人检测请求（Cache 模式）")
        
        # 获取 cache_id
        cache_id = request.form.get('cache_id', '')
        if not cache_id or cache_id not in stranger_caches:
            return jsonify({'success': False, 'error': 'Cache 不存在或已过期'}), 404
        
        cache_info = stranger_caches[cache_id]
        cache_name = cache_info['cache_name']
        cache_model = cache_info['model']
        
        # 获取视频
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '请上传视频文件'}), 400
        
        video_file = request.files['video']
        if not video_file.filename:
            return jsonify({'success': False, 'error': '视频文件名为空'}), 400
        
        # 获取查询内容
        query = request.form.get('query', '请分析这个视频中是否有陌生人。').strip()
        
        # 获取媒体分辨率参数
        media_resolution = request.form.get('media_resolution', None)
        logger.info(f"[{request_id}] Cache 模式媒体分辨率: {media_resolution}")
        
        # 读取视频
        video_bytes = video_file.read()
        video_size_mb = len(video_bytes) / 1024 / 1024
        
        # 确定视频 mime_type
        filename = video_file.filename.lower()
        mime_type_map = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.ts': 'video/mp2t'
        }
        ext = os.path.splitext(filename)[1]
        video_mime_type = mime_type_map.get(ext, 'video/mp4')
        
        logger.info(f"[{request_id}] 使用 Cache: {cache_id}, 视频大小: {video_size_mb:.2f} MB")
        
        # 构建内容
        video_part = types.Part.from_bytes(data=video_bytes, mime_type=video_mime_type)
        contents = [video_part, query]
        
        # 使用 Cache 调用 Gemini
        start_time = time.time()
        
        # 配置参数（包含媒体分辨率）
        config_params = {
            'max_output_tokens': 4096,
            'temperature': 0.4,
            'cached_content': cache_name
        }
        
        # 应用媒体分辨率设置
        if media_resolution and media_resolution in MEDIA_RESOLUTION_MAP:
            config_params["media_resolution"] = MEDIA_RESOLUTION_MAP[media_resolution]
            logger.info(f"[{request_id}] Cache 模式应用媒体分辨率: {media_resolution}")
        
        config = types.GenerateContentConfig(**config_params)
        
        response = client.models.generate_content(
            model=cache_model,
            contents=contents,
            config=config
        )
        
        processing_time = time.time() - start_time
        
        # 提取结果
        analysis_text = response.text if response.text else "分析未返回结果"
        
        # 使用智能判断函数判断是否有陌生人
        has_stranger = detect_stranger_in_analysis(analysis_text)
        logger.info(f"[{request_id}] 陌生人检测结果: has_stranger={has_stranger}")
        
        # Token 统计（包含详细信息）
        token_stats = {}
        token_details = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            token_stats = {
                'cached_content_token_count': getattr(um, 'cached_content_token_count', 0),
                'prompt_token_count': getattr(um, 'prompt_token_count', 0),
                'candidates_token_count': getattr(um, 'candidates_token_count', 0),
                'total_token_count': getattr(um, 'total_token_count', 0),
                'input_tokens': getattr(um, 'prompt_token_count', 0),
                'output_tokens': getattr(um, 'candidates_token_count', 0)
            }
            # 提取详细 token 信息
            token_details = extract_token_details(um)
        
        logger.info(f"[{request_id}] 分析完成，耗时: {processing_time:.2f}秒, Token: {token_stats}")
        
        return jsonify({
            'success': True,
            'has_stranger': has_stranger,
            'analysis': analysis_text,
            'token_stats': token_stats,
            'token_details': token_details,
            'processing_time': round(processing_time, 2),
            'cache_id': cache_id,
            'owner_names': cache_info['owner_names']
        })
        
    except Exception as e:
        logger.error(f"[{request_id}] Cache 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stranger/delete_cache', methods=['POST'])
def stranger_delete_cache():
    """删除陌生人检测 Cache"""
    try:
        data = request.get_json()
        cache_id = data.get('cache_id', '')
        
        if not cache_id or cache_id not in stranger_caches:
            return jsonify({'success': False, 'error': 'Cache 不存在'}), 404
        
        cache_info = stranger_caches[cache_id]
        cache_name = cache_info['cache_name']
        
        # 删除 Cache
        try:
            client.caches.delete(name=cache_name)
            logger.info(f"Cache 已删除: {cache_id}")
        except Exception as del_error:
            logger.warning(f"删除 Cache 时出错（可能已过期）: {del_error}")
        
        # 从本地存储移除
        del stranger_caches[cache_id]
        
        return jsonify({
            'success': True,
            'message': f'Cache {cache_id} 已删除'
        })
        
    except Exception as e:
        logger.error(f"Cache 删除失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 视频 Cache API ====================

# 存储 Cache 信息
video_caches = {}

# GCS 配置（用于 Context Caching）
GCS_BUCKET_NAME = f"{PROJECT_ID}-video-cache"

def upload_to_gcs(video_bytes: bytes, filename: str) -> str:
    """上传视频到 GCS 并返回 URI"""
    from google.cloud import storage
    
    storage_client = storage.Client(project=PROJECT_ID)
    
    # 检查 bucket 是否存在，不存在则创建
    try:
        bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
    except Exception:
        logger.info(f"创建 GCS bucket: {GCS_BUCKET_NAME}")
        bucket = storage_client.create_bucket(GCS_BUCKET_NAME, location="us-central1")
    
    # 生成唯一文件名
    blob_name = f"cache-videos/{int(time.time())}_{filename}"
    blob = bucket.blob(blob_name)
    
    # 上传
    blob.upload_from_string(video_bytes, content_type='video/mp4')
    
    gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    logger.info(f"视频已上传到 GCS: {gcs_uri}")
    
    return gcs_uri


def scan_video_for_owners(video_bytes: bytes, person_embeddings: dict, target_fps: float = 2.0) -> dict:
    """使用 InsightFace 扫描视频，识别 Owner 出场时间"""
    import cv2
    import numpy as np
    import tempfile
    
    owner_appearances = {}  # {name: [{'time': '00:05', 'frame': 150}, ...]}
    
    # 保存视频到临时文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.error("无法打开视频文件")
            return owner_appearances
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / target_fps))
        
        logger.info(f"扫描视频: fps={fps}, 总帧数={total_frames}, 采样间隔={frame_interval}")
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 按采样率处理
            if frame_count % frame_interval != 0:
                continue
            
            processed_count += 1
            
            # 使用 InsightFace 检测人脸
            try:
                fa = insightface_utils.get_face_analysis()
                if fa is None:
                    continue
                faces = fa.get(frame)
                
                for face in faces:
                    face_embedding = face.embedding
                    
                    # 匹配 Owner
                    for person_name, person_embedding in person_embeddings.items():
                        similarity = np.dot(face_embedding, person_embedding) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(person_embedding)
                        )
                        
                        if similarity > 0.5:  # 匹配阈值
                            # 计算时间
                            seconds = frame_count / fps
                            time_str = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
                            
                            if person_name not in owner_appearances:
                                owner_appearances[person_name] = []
                            
                            # 避免重复记录相近时间
                            if not owner_appearances[person_name] or \
                               abs(seconds - owner_appearances[person_name][-1].get('seconds', 0)) > 2:
                                owner_appearances[person_name].append({
                                    'time': time_str,
                                    'frame': frame_count,
                                    'seconds': seconds,
                                    'similarity': float(similarity)
                                })
            except Exception as e:
                logger.warning(f"帧 {frame_count} 处理失败: {e}")
                continue
        
        cap.release()
        logger.info(f"视频扫描完成: 处理 {processed_count} 帧, 识别到 {len(owner_appearances)} 个 Owner")
        
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    return owner_appearances


@app.route('/api/cache/create', methods=['POST'])
def cache_create():
    """创建视频 Cache（通过 GCS 上传，集成 InsightFace 人脸识别）"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '未上传视频'}), 400
        
        video_file = request.files['video']
        owner_data = request.form.get('owner_data', '')  # JSON 格式的 Owner 照片数据
        target_fps = float(request.form.get('target_fps', 2.0))
        base_prompt = request.form.get('base_prompt', '')  # 用户自定义的基础分析提示词
        
        # 调试日志
        logger.info(f"收到 owner_data 长度: {len(owner_data) if owner_data else 0}")
        logger.info(f"owner_data 前200字符: {owner_data[:200] if owner_data else 'None'}")
        
        if not video_file.filename:
            return jsonify({'success': False, 'error': '视频文件名为空'}), 400
        
        # 读取视频数据
        video_bytes = video_file.read()
        video_size_mb = len(video_bytes) / 1024 / 1024
        logger.info(f"创建 Cache: 视频大小={video_size_mb:.2f} MB")
        
        # 确定 mime_type
        filename = video_file.filename.lower()
        mime_type_map = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        ext = os.path.splitext(filename)[1]
        video_mime_type = mime_type_map.get(ext, 'video/mp4')
        
        # 处理 Owner 照片，提取人脸特征
        person_embeddings = {}
        owner_names = []
        owner_appearances = {}
        owner_details = {}  # 详细状态: {name: {'has_photo': bool, 'face_extracted': bool, 'recognized_in_video': bool, 'appearances': int}}
        
        if owner_data:
            try:
                owners = json.loads(owner_data)
                logger.info(f"收到 {len(owners)} 个 Owner 数据")
                
                for owner in owners:
                    name = owner.get('name')
                    image_data = owner.get('imageData')
                    # 调试日志
                    logger.info(f"Owner '{name}': imageData存在={bool(image_data)}, 长度={len(image_data) if image_data else 0}")
                    
                    if name:
                        owner_names.append(name)
                        owner_details[name] = {
                            'has_photo': bool(image_data),
                            'face_extracted': False,
                            'recognized_in_video': False,
                            'appearances': 0,
                            'status': '未上传照片'
                        }
                        
                        if image_data:
                            # 解码图片并提取人脸特征
                            if image_data.startswith('data:'):
                                image_data = image_data.split(',')[1]
                            
                            import cv2
                            import numpy as np
                            
                            img_bytes = base64.b64decode(image_data)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if img is not None:
                                # 使用 insightface_utils 获取 FaceAnalysis 实例
                                fa = insightface_utils.get_face_analysis()
                                if fa is not None:
                                    faces = fa.get(img)
                                    if faces:
                                        person_embeddings[name] = faces[0].embedding
                                        owner_details[name]['face_extracted'] = True
                                        owner_details[name]['status'] = '已提取人脸特征'
                                        logger.info(f"提取 {name} 的人脸特征成功")
                                    else:
                                        owner_details[name]['status'] = '照片中未检测到人脸'
                                        logger.warning(f"未在 {name} 的照片中检测到人脸")
                                else:
                                    owner_details[name]['status'] = 'InsightFace 未初始化'
                                    logger.error("InsightFace 模型未初始化")
                            else:
                                owner_details[name]['status'] = '照片解码失败'
                        
            except Exception as e:
                logger.error(f"处理 Owner 照片失败: {e}")
        
        # 如果有 Owner 特征，扫描视频识别出场时间
        if person_embeddings:
            logger.info("开始使用 InsightFace 扫描视频...")
            owner_appearances = scan_video_for_owners(video_bytes, person_embeddings, target_fps)
            
            # 更新 owner_details 的识别状态
            for name in owner_details:
                if name in owner_appearances and owner_appearances[name]:
                    owner_details[name]['recognized_in_video'] = True
                    owner_details[name]['appearances'] = len(owner_appearances[name])
                    owner_details[name]['status'] = f'视频中出现 {len(owner_appearances[name])} 次'
                elif owner_details[name]['face_extracted']:
                    owner_details[name]['status'] = '视频中未检测到此人'
        
        # 构建系统指令（包含 InsightFace 识别结果）
        system_instruction = """你是一个专业的视频分析助手。

【任务】
分析视频内容，识别人物、动作和场景。

【人物标注规则】
- Owner: 已通过人脸识别确认的人物，请在描述中使用其名字并标注 "(Owner)"
- 未标注的人物请称为"陌生人"或"其他人"
"""
        
        # 添加 InsightFace 识别结果
        if owner_appearances:
            system_instruction += "\n【InsightFace 人脸识别结果】\n"
            system_instruction += "以下是通过人脸识别技术确认的已标注人物 (Owner) 及其出现时间：\n"
            for name, appearances in owner_appearances.items():
                time_points = [app['time'] for app in appearances[:10]]  # 最多显示10个时间点
                system_instruction += f"  - {name} (Owner): 出现于 {', '.join(time_points)}\n"
        elif owner_names:
            system_instruction += f"\n【已标注人物】\n本视频中已标注的 Owner: {', '.join(owner_names)}\n"
            system_instruction += "（注：未能在视频中匹配到这些人物的人脸）\n"
        
        # 添加用户自定义的基础分析提示词
        if base_prompt:
            system_instruction += f"\n【用户分析要求】\n{base_prompt}\n"
            logger.info(f"添加用户自定义提示词: {base_prompt[:50]}...")
        
        # 上传视频到 GCS
        logger.info("正在上传视频到 GCS...")
        try:
            gcs_uri = upload_to_gcs(video_bytes, video_file.filename)
        except Exception as gcs_error:
            logger.error(f"GCS 上传失败: {gcs_error}")
            return jsonify({
                'success': False,
                'error': f'GCS 上传失败: {str(gcs_error)}',
                'hint': '请确保已配置 Google Cloud Storage 权限'
            }), 500
        
        # 使用 file_uri 创建视频 Part
        video_part = types.Part.from_uri(file_uri=gcs_uri, mime_type=video_mime_type)
        
        # 创建 Cache - 使用用户选择的模型
        cache_model = request.form.get('model_id', DEFAULT_MODEL_ID)
        
        try:
            logger.info("正在创建 Context Cache...")
            cache = client.caches.create(
                model=cache_model,
                config=types.CreateCachedContentConfig(
                    display_name=f"video-cache-{int(time.time())}",
                    system_instruction=system_instruction,
                    contents=[video_part],
                    ttl="3600s"  # 1小时有效期
                )
            )
            
            cache_id = cache.name.split('/')[-1] if '/' in cache.name else cache.name
            
            # 存储 Cache 信息
            video_caches[cache_id] = {
                'cache_name': cache.name,
                'model': cache_model,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'expires_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 3600)),
                'owner_names': owner_names,
                'owner_appearances': {k: len(v) for k, v in owner_appearances.items()},
                'video_size': len(video_bytes),
                'gcs_uri': gcs_uri
            }
            
            logger.info(f"Cache 创建成功: {cache_id}")
            
            return jsonify({
                'success': True,
                'cache_id': cache_id,
                'cache_name': cache.name,
                'model': cache_model,
                'created_at': video_caches[cache_id]['created_at'],
                'expires_at': video_caches[cache_id]['expires_at'],
                'gcs_uri': gcs_uri,
                'owner_recognition': {
                    'owners_count': len(owner_names),
                    'recognized_count': len(owner_appearances),
                    'appearances': {k: len(v) for k, v in owner_appearances.items()},
                    'details': owner_details  # 每个Owner的详细状态
                },
                'usage_metadata': {
                    'total_token_count': getattr(cache.usage_metadata, 'total_token_count', 0) if hasattr(cache, 'usage_metadata') else 0
                }
            })
            
        except Exception as cache_error:
            logger.error(f"Cache 创建失败: {cache_error}")
            return jsonify({
                'success': False, 
                'error': f'Cache 创建失败: {str(cache_error)}',
                'hint': 'Context Caching 需要特定模型支持，请确认模型和区域配置'
            }), 500
        
    except Exception as e:
        logger.error(f"Cache 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cache/query', methods=['POST'])
def cache_query():
    """使用 Cache 查询分析"""
    try:
        data = request.get_json()
        cache_id = data.get('cache_id')
        query = data.get('query', '请分析视频内容')
        
        if not cache_id:
            return jsonify({'success': False, 'error': '缺少 cache_id'}), 400
        
        if cache_id not in video_caches:
            return jsonify({'success': False, 'error': 'Cache 不存在或已过期'}), 404
        
        cache_info = video_caches[cache_id]
        cache_name = cache_info['cache_name']
        cache_model = cache_info['model']
        
        logger.info(f"使用 Cache 查询: {cache_id}, query={query[:50]}...")
        
        # 使用 Cache 进行查询
        config = types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.4,
            cached_content=cache_name
        )
        
        response = client.models.generate_content(
            model=cache_model,
            contents=[query],
            config=config
        )
        
        # 处理结果
        analysis = ''
        if response and response.text:
            analysis = response.text.strip()
        else:
            analysis = '分析未返回结果'
        
        # Token 统计
        token_stats = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            token_stats = {
                'cached_content_token_count': getattr(um, 'cached_content_token_count', 0),
                'prompt_token_count': getattr(um, 'prompt_token_count', 0),
                'candidates_token_count': getattr(um, 'candidates_token_count', 0),
                'total_token_count': getattr(um, 'total_token_count', 0)
            }
        
        logger.info(f"Cache 查询完成, Token: {token_stats}")
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'token_stats': token_stats,
            'cache_id': cache_id
        })
        
    except Exception as e:
        logger.error(f"Cache 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cache/delete', methods=['POST'])
def cache_delete():
    """删除 Cache 和 GCS 文件"""
    try:
        data = request.get_json()
        cache_id = data.get('cache_id')
        
        if not cache_id:
            return jsonify({'success': False, 'error': '缺少 cache_id'}), 400
        
        if cache_id not in video_caches:
            return jsonify({'success': False, 'error': 'Cache 不存在'}), 404
        
        cache_info = video_caches[cache_id]
        cache_name = cache_info['cache_name']
        gcs_uri = cache_info.get('gcs_uri')
        
        # 删除 Cache
        try:
            client.caches.delete(name=cache_name)
            logger.info(f"Cache 已删除: {cache_id}")
        except Exception as del_error:
            logger.warning(f"删除 Cache 时出错（可能已过期）: {del_error}")
        
        # 删除 GCS 文件
        if gcs_uri:
            try:
                from google.cloud import storage
                storage_client = storage.Client(project=PROJECT_ID)
                # 解析 gcs_uri: gs://bucket-name/path/to/file
                parts = gcs_uri.replace("gs://", "").split("/", 1)
                bucket_name = parts[0]
                blob_name = parts[1] if len(parts) > 1 else ""
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.delete()
                logger.info(f"GCS 文件已删除: {gcs_uri}")
            except Exception as gcs_error:
                logger.warning(f"删除 GCS 文件时出错: {gcs_error}")
        
        # 从本地存储移除
        del video_caches[cache_id]
        
        return jsonify({
            'success': True,
            'message': f'Cache {cache_id} 已删除'
        })
        
    except Exception as e:
        logger.error(f"Cache 删除失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cache/list', methods=['GET'])
def cache_list():
    """列出所有 Cache"""
    try:
        cache_list = []
        for cache_id, info in video_caches.items():
            cache_list.append({
                'cache_id': cache_id,
                'created_at': info['created_at'],
                'expires_at': info['expires_at'],
                'model': info['model'],
                'video_size': info['video_size']
            })
        
        return jsonify({
            'success': True,
            'caches': cache_list
        })
        
    except Exception as e:
        logger.error(f"列出 Cache 失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info("=" * 60)
    logger.info("Gemini 2.5 Flash Lite 图片推理服务启动")
    logger.info("=" * 60)
    logger.info(f"端口: {port}")
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"默认模型: {DEFAULT_MODEL_ID}")
    logger.info(f"可用模型: {list(AVAILABLE_MODELS.keys())}")
    logger.info(f"Location: {LOCATION}")
    logger.info(f"LangChain 可用: {LANGCHAIN_AVAILABLE}")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=True)
