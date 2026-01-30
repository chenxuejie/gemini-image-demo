"""
使用 ChatVertexAI 分析图片并显示详细的 Token 信息
包括 image token, text token 等详细信息
"""

import base64
import os
from typing import Optional
from pydantic import Field
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

# 配置
PROJECT_ID = "cloud-llm-preview1"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-001"
IMAGE_PATH = "image.png"


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
            print(f"[ChatVertexAIWithMediaResolution] 注入 media_resolution={enum_value}")
        
        return super().invoke(input, config, stop=stop, **kwargs)


def analyze_image_with_token_details(image_path: str, prompt: str, media_resolution: str = 'medium'):
    """
    使用 LangChain ChatVertexAI 分析图片并打印详细的 Token 信息
    
    Args:
        image_path: 图片文件路径
        prompt: 分析提示词
        media_resolution: 媒体分辨率 (low, medium, high)
    """
    print("=" * 80)
    print("Gemini 图片分析 - Token 详细信息")
    print("=" * 80)
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return
    
    print(f"\n图片路径: {image_path}")
    print(f"模型: {MODEL_ID}")
    print(f"Media Resolution: {media_resolution}")
    print(f"Prompt: {prompt}")
    
    # 读取图片并转换为 base64
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    file_size_kb = len(image_bytes) / 1024
    print(f"图片大小: {file_size_kb:.2f} KB")
    
    # 确定 MIME 类型
    if image_path.lower().endswith('.png'):
        mime_type = 'image/png'
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = 'image/jpeg'
    elif image_path.lower().endswith('.gif'):
        mime_type = 'image/gif'
    elif image_path.lower().endswith('.webp'):
        mime_type = 'image/webp'
    else:
        mime_type = 'image/png'
    
    print(f"MIME 类型: {mime_type}")
    
    print("\n" + "-" * 80)
    print("开始调用 Gemini API...")
    print("-" * 80)
    
    try:
        # 创建 ChatVertexAIWithMediaResolution 实例
        llm = ChatVertexAIWithMediaResolution(
            model=MODEL_ID,
            project=PROJECT_ID,
            location=LOCATION,
            default_media_resolution=media_resolution
        )
        
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
        
        # 调用模型
        response = llm.invoke([message])
        
        print("\n" + "=" * 80)
        print("Token 使用详情")
        print("=" * 80)
        
        # 提取 usage_metadata
        usage_metadata = getattr(response, 'usage_metadata', None)
        
        if usage_metadata:
            print(f"\nInput Tokens (总计): {usage_metadata.get('input_tokens', 'N/A')}")
            print(f"Output Tokens (总计): {usage_metadata.get('output_tokens', 'N/A')}")
            print(f"Total Tokens: {usage_metadata.get('total_tokens', 'N/A')}")
            
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
                print("\nInput Tokens 详情:")
                print("-" * 40)
                modality_names = {1: 'TEXT', 2: 'IMAGE', 3: 'VIDEO', 4: 'AUDIO'}
                for detail in prompt_tokens_details:
                    modality = detail.get('modality', 0)
                    token_count = detail.get('token_count', 0)
                    modality_name = modality_names.get(modality, f'UNKNOWN({modality})')
                    if modality == 2:  # IMAGE
                        print(f"Image Tokens: {token_count}")
                    elif modality == 1:  # TEXT
                        print(f"Text Tokens: {token_count}")
                    elif modality == 3:  # VIDEO
                        print(f"Video Tokens: {token_count}")
                    elif modality == 4:  # AUDIO
                        print(f"Audio Tokens: {token_count}")
                    else:
                        print(f"  ❓ {modality_name} Tokens: {token_count}")
            
            # Candidates tokens 详情
            candidates_tokens_details = detailed_usage.get('candidates_tokens_details', [])
            if candidates_tokens_details:
                print("\nOutput Tokens 详情:")
                print("-" * 40)
                for detail in candidates_tokens_details:
                    modality = detail.get('modality', 0)
                    token_count = detail.get('token_count', 0)
                    modality_name = modality_names.get(modality, f'UNKNOWN({modality})')
                    if modality == 2:  # IMAGE
                        print(f"Image Tokens: {token_count}")
                    elif modality == 1:  # TEXT
                        print(f"Text Tokens: {token_count}")
                    elif modality == 3:  # VIDEO
                        print(f"Video Tokens: {token_count}")
                    elif modality == 4:  # AUDIO
                        print(f"Audio Tokens: {token_count}")
                    else:
                        print(f"{modality_name} Tokens: {token_count}")
            
            # 其他 token 信息
            print("\n其他 Token 信息:")
            print("-" * 40)
            print(f"Thinking Tokens: {detailed_usage.get('thoughts_token_count', 0)}")
            print(f"Cached Content Tokens: {detailed_usage.get('cached_content_token_count', 0)}")
            
            # Cache tokens 详情
            cache_tokens_details = detailed_usage.get('cache_tokens_details', [])
            if cache_tokens_details:
                print(f"Cache Tokens Details: {cache_tokens_details}")
            
            # 原始 token 计数
            print("\n原始 Token 计数:")
            print("-" * 40)
            print(f"prompt_token_count: {detailed_usage.get('prompt_token_count', 0)}")
            print(f"candidates_token_count: {detailed_usage.get('candidates_token_count', 0)}")
            print(f"total_token_count: {detailed_usage.get('total_token_count', 0)}")
        
        # 打印 AI 响应内容
        print("\n" + "=" * 80)
        print("AI 响应内容")
        print("=" * 80)
        print(response.content if hasattr(response, 'content') else str(response))
        
        print("\n" + "=" * 80)
        print("分析完成!")
        print("=" * 80)
        
        return response
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, IMAGE_PATH)
    
    # 分析图片
    analyze_image_with_token_details(
        image_path=image_path,
        prompt="请详细描述这张图片的内容",
        media_resolution='low'
    )


if __name__ == "__main__":
    main()
