"""
使用 ChatVertexAI 分析视频并显示详细的 Token 信息
包括 video token, text token 等详细信息，以及处理时间
"""

import base64
import os
import time
from typing import Optional
from pydantic import Field
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

# 配置
PROJECT_ID = "cloud-llm-preview1"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash"
VIDEO_PATH = "1.ts"


class ChatVertexAIWithMediaResolution(ChatVertexAI):
    """
    继承 ChatVertexAI，添加 media_resolution 支持。
    """
    
    default_media_resolution: Optional[str] = Field(default=None)
    
    def invoke(self, input, config=None, *, stop=None, **kwargs):
        """重写 invoke 方法，注入 media_resolution 参数"""
        if 'media_resolution' not in kwargs and self.default_media_resolution:
            resolution_map = {
                'low': 'MEDIA_RESOLUTION_LOW',
                'medium': 'MEDIA_RESOLUTION_MEDIUM',
                'high': 'MEDIA_RESOLUTION_HIGH',
                'unspecified': 'MEDIA_RESOLUTION_UNSPECIFIED'
            }
            enum_value = resolution_map.get(self.default_media_resolution, self.default_media_resolution)
            kwargs['media_resolution'] = enum_value
            print(f"[ChatVertexAIWithMediaResolution] 注入 media_resolution={enum_value}")
        
        return super().invoke(input, config, stop=stop, **kwargs)


def analyze_video_with_token_details(video_path: str, prompt: str, media_resolution: str = 'medium'):
    """
    使用 LangChain ChatVertexAI 分析视频并打印详细的 Token 信息和处理时间
    
    Args:
        video_path: 视频文件路径
        prompt: 分析提示词
        media_resolution: 媒体分辨率 (low, medium, high)
    """
    print("=" * 80)
    print("Gemini 视频分析 - Token 详细信息")
    print("=" * 80)
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    print(f"\n视频路径: {video_path}")
    print(f"模型: {MODEL_ID}")
    print(f"Media Resolution: {media_resolution}")
    print(f"Prompt: {prompt}")
    
    # 读取视频并转换为 base64
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    file_size_kb = len(video_bytes) / 1024
    file_size_mb = file_size_kb / 1024
    print(f"视频大小: {file_size_kb:.2f} KB ({file_size_mb:.2f} MB)")
    
    # 确定 MIME 类型
    if video_path.lower().endswith('.mp4'):
        mime_type = 'video/mp4'
    elif video_path.lower().endswith('.ts'):
        mime_type = 'video/mp2t'
    elif video_path.lower().endswith('.mov'):
        mime_type = 'video/quicktime'
    elif video_path.lower().endswith('.avi'):
        mime_type = 'video/x-msvideo'
    elif video_path.lower().endswith('.webm'):
        mime_type = 'video/webm'
    elif video_path.lower().endswith('.mkv'):
        mime_type = 'video/x-matroska'
    else:
        mime_type = 'video/mp4'
    
    print(f"MIME 类型: {mime_type}")
    
    print("\n" + "-" * 80)
    print("开始调用 Gemini API...")
    print("-" * 80)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 创建 ChatVertexAIWithMediaResolution 实例
        llm = ChatVertexAIWithMediaResolution(
            model=MODEL_ID,
            project=PROJECT_ID,
            location=LOCATION,
            default_media_resolution=media_resolution
        )
        
        # 创建包含视频的消息
        message = HumanMessage(
            content=[
                {
                    "type": "media",
                    "mime_type": mime_type,
                    "data": video_base64
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )
        
        # 调用模型
        response = llm.invoke([message])
        
        # 记录结束时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("处理时间")
        print("=" * 80)
        print(f"总处理时间: {processing_time:.2f} 秒")
        
        print("\n" + "=" * 80)
        print("Token 使用详情")
        print("=" * 80)
        
        # 提取 usage_metadata
        usage_metadata = getattr(response, 'usage_metadata', None)
        
        if usage_metadata:
            print(f"\n📊 Input Tokens (总计): {usage_metadata.get('input_tokens', 'N/A')}")
            print(f"📊 Output Tokens (总计): {usage_metadata.get('output_tokens', 'N/A')}")
            print(f"📊 Total Tokens: {usage_metadata.get('total_tokens', 'N/A')}")
            
            # 打印详细的 token 信息
            print("\n" + "-" * 40)
            print("完整 usage_metadata 内容:")
            print("-" * 40)
            for key, value in usage_metadata.items():
                print(f"  {key}: {value}")
        else:
            print("未获取到 usage_metadata")
        
        # 从 response_metadata 中提取更详细的 token 信息
        response_metadata = getattr(response, 'response_metadata', None)
        if response_metadata and 'usage_metadata' in response_metadata:
            detailed_usage = response_metadata['usage_metadata']
            
            print("\n" + "=" * 80)
            print("详细 Token 分解 (来自 response_metadata)")
            print("=" * 80)
            
            # Prompt tokens 详情
            prompt_tokens_details = detailed_usage.get('prompt_tokens_details', [])
            if prompt_tokens_details:
                print("\n📥 Input Tokens 详情:")
                print("-" * 40)
                modality_names = {1: 'TEXT', 2: 'IMAGE', 3: 'VIDEO', 4: 'AUDIO'}
                for detail in prompt_tokens_details:
                    modality = detail.get('modality', 0)
                    token_count = detail.get('token_count', 0)
                    modality_name = modality_names.get(modality, f'UNKNOWN({modality})')
                    if modality == 2:  # IMAGE
                        print(f"  🖼️  Image Tokens: {token_count}")
                    elif modality == 1:  # TEXT
                        print(f"  📝 Text Tokens: {token_count}")
                    elif modality == 3:  # VIDEO
                        print(f"  🎬 Video Tokens: {token_count}")
                    elif modality == 4:  # AUDIO
                        print(f"  🔊 Audio Tokens: {token_count}")
                    else:
                        print(f"  ❓ {modality_name} Tokens: {token_count}")
            
            # Candidates tokens 详情
            candidates_tokens_details = detailed_usage.get('candidates_tokens_details', [])
            if candidates_tokens_details:
                print("\n📤 Output Tokens 详情:")
                print("-" * 40)
                for detail in candidates_tokens_details:
                    modality = detail.get('modality', 0)
                    token_count = detail.get('token_count', 0)
                    modality_name = modality_names.get(modality, f'UNKNOWN({modality})')
                    if modality == 2:  # IMAGE
                        print(f"  🖼️  Image Tokens: {token_count}")
                    elif modality == 1:  # TEXT
                        print(f"  📝 Text Tokens: {token_count}")
                    elif modality == 3:  # VIDEO
                        print(f"  🎬 Video Tokens: {token_count}")
                    elif modality == 4:  # AUDIO
                        print(f"  🔊 Audio Tokens: {token_count}")
                    else:
                        print(f"  ❓ {modality_name} Tokens: {token_count}")
            
            # 其他 token 信息
            print("\n📋 其他 Token 信息:")
            print("-" * 40)
            print(f"  💭 Thinking Tokens: {detailed_usage.get('thoughts_token_count', 0)}")
            print(f"  📦 Cached Content Tokens: {detailed_usage.get('cached_content_token_count', 0)}")
            
            # Cache tokens 详情
            cache_tokens_details = detailed_usage.get('cache_tokens_details', [])
            if cache_tokens_details:
                print(f"  Cache Tokens Details: {cache_tokens_details}")
            
            # 原始 token 计数
            print("\n📊 原始 Token 计数:")
            print("-" * 40)
            print(f"  prompt_token_count: {detailed_usage.get('prompt_token_count', 0)}")
            print(f"  candidates_token_count: {detailed_usage.get('candidates_token_count', 0)}")
            print(f"  total_token_count: {detailed_usage.get('total_token_count', 0)}")
        
        # 打印 AI 响应内容
        print("\n" + "=" * 80)
        print("AI 响应内容")
        print("=" * 80)
        print(response.content if hasattr(response, 'content') else str(response))
        
        print("\n" + "=" * 80)
        print(f"分析完成! 总处理时间: {processing_time:.2f} 秒")
        print("=" * 80)
        
        return response
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\n❌ 错误 (处理时间: {processing_time:.2f} 秒): {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, VIDEO_PATH)
    
    # 分析视频
    analyze_video_with_token_details(
        video_path=video_path,
        prompt="请详细描述这个视频的内容，包括场景、人物、动作和任何重要的事件。",
        media_resolution='medium'
    )


if __name__ == "__main__":
    main()
