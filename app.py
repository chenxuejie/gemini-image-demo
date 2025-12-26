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
from flask import Flask, request, jsonify, send_from_directory
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
PROJECT_ID = "cloud-llm-preview1"
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
