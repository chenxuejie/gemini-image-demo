"""
直接使用 google-genai SDK 分析视频并显示详细的 Token 信息
不使用 LangChain，最快的调用方式，关闭 thinking 模式
"""

import base64
import os
import time
from google import genai
from google.genai import types

# 配置
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash"
VIDEO_PATH = "1.ts"


def analyze_video_with_token_details(video_path: str, prompt: str):
    """
    使用 google-genai SDK 分析视频并打印详细的 Token 信息和处理时间
    
    Args:
        video_path: 视频文件路径
        prompt: 分析提示词
    """
    print("=" * 80)
    print("Gemini 视频分析 - Token 详细信息 (Direct API)")
    print("=" * 80)
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    print(f"\n视频路径: {video_path}")
    print(f"模型: {MODEL_ID}")
    print(f"Thinking: 关闭")
    print(f"Prompt: {prompt}")
    
    # 读取视频并转换为 base64
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    
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
        # 创建 genai 客户端
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )
        
        # 创建视频内容
        video_part = types.Part.from_bytes(
            data=video_bytes,
            mime_type=mime_type
        )
        
        # 配置生成参数 - 关闭 thinking 模式，设置 media resolution 为 low
        generate_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0  # 关闭 thinking
            ),
            media_resolution="MEDIA_RESOLUTION_MEDIUM"  # 设置 media resolution 为 medium
        )
        
        # 调用模型
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[video_part, prompt],
            config=generate_config
        )
        
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
        usage_metadata = response.usage_metadata
        
        if usage_metadata:
            print(f"\n📊 Input Tokens (总计): {usage_metadata.prompt_token_count}")
            print(f"📊 Output Tokens (总计): {usage_metadata.candidates_token_count}")
            print(f"📊 Total Tokens: {usage_metadata.total_token_count}")
            
            # 打印 Thinking Tokens
            if hasattr(usage_metadata, 'thoughts_token_count'):
                print(f"💭 Thinking Tokens: {usage_metadata.thoughts_token_count}")
            
            # 打印 Cached Content Tokens
            if hasattr(usage_metadata, 'cached_content_token_count'):
                print(f"📦 Cached Content Tokens: {usage_metadata.cached_content_token_count}")
            
            # 打印详细的 token 信息
            print("\n" + "-" * 40)
            print("Input Tokens 详情:")
            print("-" * 40)
            
            if hasattr(usage_metadata, 'prompt_tokens_details') and usage_metadata.prompt_tokens_details:
                modality_names = {1: 'TEXT', 2: 'IMAGE', 3: 'VIDEO', 4: 'AUDIO'}
                for detail in usage_metadata.prompt_tokens_details:
                    modality = detail.modality if hasattr(detail, 'modality') else 0
                    token_count = detail.token_count if hasattr(detail, 'token_count') else 0
                    if modality == 2:
                        print(f"  🖼️  Image Tokens: {token_count}")
                    elif modality == 1:
                        print(f"  📝 Text Tokens: {token_count}")
                    elif modality == 3:
                        print(f"  🎬 Video Tokens: {token_count}")
                    elif modality == 4:
                        print(f"  🔊 Audio Tokens: {token_count}")
                    else:
                        modality_name = modality_names.get(modality, f'UNKNOWN({modality})')
                        print(f"  ❓ {modality_name} Tokens: {token_count}")
            
            print("\n" + "-" * 40)
            print("Output Tokens 详情:")
            print("-" * 40)
            
            if hasattr(usage_metadata, 'candidates_tokens_details') and usage_metadata.candidates_tokens_details:
                for detail in usage_metadata.candidates_tokens_details:
                    modality = detail.modality if hasattr(detail, 'modality') else 0
                    token_count = detail.token_count if hasattr(detail, 'token_count') else 0
                    if modality == 2:
                        print(f"  🖼️  Image Tokens: {token_count}")
                    elif modality == 1:
                        print(f"  📝 Text Tokens: {token_count}")
                    elif modality == 3:
                        print(f"  🎬 Video Tokens: {token_count}")
                    elif modality == 4:
                        print(f"  🔊 Audio Tokens: {token_count}")
                    else:
                        modality_name = modality_names.get(modality, f'UNKNOWN({modality})')
                        print(f"  ❓ {modality_name} Tokens: {token_count}")
        else:
            print("未获取到 usage_metadata")
        
        # 打印 AI 响应内容
        print("\n" + "=" * 80)
        print("AI 响应内容")
        print("=" * 80)
        print(response.text)
        
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
        prompt="请详细描述这个视频的内容，包括场景、人物、动作和任何重要的事件。"
    )


if __name__ == "__main__":
    main()
