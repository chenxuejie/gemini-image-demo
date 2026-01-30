"""
InsightFace 人脸识别工具模块
提供人脸检测、特征提取、人脸比对等功能
"""

import os
import json
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image
import io

# 配置日志
logger = logging.getLogger(__name__)

# InsightFace 延迟导入（避免启动时报错）
_face_analysis = None
_model_initialized = False


def get_face_analysis():
    """获取 FaceAnalysis 实例（延迟初始化）"""
    global _face_analysis, _model_initialized
    
    if not _model_initialized:
        try:
            from insightface.app import FaceAnalysis
            
            # 初始化模型，使用 buffalo_l 模型（精度较高）
            _face_analysis = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']  # 使用 CPU
            )
            # 设置检测尺寸
            _face_analysis.prepare(ctx_id=0, det_size=(640, 640))
            _model_initialized = True
            logger.info("InsightFace 模型初始化成功")
        except Exception as e:
            logger.error(f"InsightFace 模型初始化失败: {e}")
            _face_analysis = None
            _model_initialized = True  # 标记为已尝试初始化
    
    return _face_analysis


def detect_faces(image_data: bytes) -> List[Dict]:
    """
    检测图片中的人脸
    
    Args:
        image_data: 图片字节数据
    
    Returns:
        人脸列表，每个人脸包含 bbox, embedding 等信息
    """
    fa = get_face_analysis()
    if fa is None:
        logger.error("InsightFace 模型未初始化")
        return []
    
    try:
        # 将字节数据转换为 numpy 数组
        image = Image.open(io.BytesIO(image_data))
        # 转换为 RGB（InsightFace 需要 BGR，但这里我们用 RGB 再转换）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image)
        # InsightFace 使用 BGR 格式
        img_bgr = img_array[:, :, ::-1]
        
        # 检测人脸
        faces = fa.get(img_bgr)
        
        results = []
        for i, face in enumerate(faces):
            result = {
                'index': i,
                'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
                'det_score': float(face.det_score) if hasattr(face, 'det_score') else 0,
                'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None,
            }
            results.append(result)
        
        logger.info(f"检测到 {len(results)} 个人脸")
        return results
        
    except Exception as e:
        logger.error(f"人脸检测失败: {e}")
        return []


def extract_embedding(image_data: bytes) -> Optional[np.ndarray]:
    """
    从图片中提取人脸特征向量（仅提取第一个检测到的人脸）
    
    Args:
        image_data: 图片字节数据
    
    Returns:
        512 维特征向量，如果未检测到人脸则返回 None
    """
    faces = detect_faces(image_data)
    if not faces:
        return None
    
    # 返回第一个人脸的特征向量
    embedding = faces[0].get('embedding')
    if embedding:
        return np.array(embedding)
    return None


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    计算两个特征向量的相似度（余弦相似度）
    
    Args:
        emb1: 特征向量1
        emb2: 特征向量2
    
    Returns:
        相似度分数 (0-1)
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    # 归一化
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    
    # 计算余弦相似度
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # 转换到 0-1 范围
    similarity = (similarity + 1) / 2
    
    return float(similarity)


# ==================== 户主数据库管理 ====================

# 默认数据目录
DEFAULT_FACE_DB_DIR = os.path.join(os.path.dirname(__file__), 'face_db')
DEFAULT_EMBEDDINGS_DIR = os.path.join(DEFAULT_FACE_DB_DIR, 'embeddings')
DEFAULT_FACES_JSON = os.path.join(DEFAULT_FACE_DB_DIR, 'faces.json')


def ensure_face_db_dirs():
    """确保人脸数据库目录存在"""
    os.makedirs(DEFAULT_FACE_DB_DIR, exist_ok=True)
    os.makedirs(DEFAULT_EMBEDDINGS_DIR, exist_ok=True)


def load_face_db() -> Dict[str, Dict]:
    """
    加载户主数据库
    
    Returns:
        户主字典 {name: {name, photo_count, created_at}}
    """
    ensure_face_db_dirs()
    
    if not os.path.exists(DEFAULT_FACES_JSON):
        return {}
    
    try:
        with open(DEFAULT_FACES_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载户主数据库失败: {e}")
        return {}


def save_face_db(face_db: Dict[str, Dict]):
    """保存户主数据库"""
    ensure_face_db_dirs()
    
    try:
        with open(DEFAULT_FACES_JSON, 'w', encoding='utf-8') as f:
            json.dump(face_db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存户主数据库失败: {e}")


def register_face(name: str, image_data: bytes) -> Dict:
    """
    注册户主人脸
    
    Args:
        name: 户主姓名
        image_data: 照片字节数据
    
    Returns:
        注册结果 {success, message, name, embedding_file}
    """
    ensure_face_db_dirs()
    
    # 检测并提取人脸特征
    embedding = extract_embedding(image_data)
    if embedding is None:
        return {
            'success': False,
            'message': '未检测到人脸，请上传包含清晰人脸的照片'
        }
    
    # 保存特征向量
    embedding_file = os.path.join(DEFAULT_EMBEDDINGS_DIR, f'{name}.npy')
    
    # 如果已存在，加载现有特征并取平均（多张照片取平均可以提高识别率）
    if os.path.exists(embedding_file):
        try:
            existing_embedding = np.load(embedding_file)
            # 取平均
            embedding = (existing_embedding + embedding) / 2
            # 重新归一化
            embedding = embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.warning(f"加载现有特征失败，使用新特征: {e}")
    
    np.save(embedding_file, embedding)
    
    # 更新数据库
    face_db = load_face_db()
    if name in face_db:
        face_db[name]['photo_count'] = face_db[name].get('photo_count', 1) + 1
    else:
        from datetime import datetime
        face_db[name] = {
            'name': name,
            'photo_count': 1,
            'created_at': datetime.now().isoformat()
        }
    save_face_db(face_db)
    
    return {
        'success': True,
        'message': f'户主 {name} 注册成功',
        'name': name,
        'photo_count': face_db[name]['photo_count']
    }


def get_all_faces() -> List[Dict]:
    """获取所有已注册的户主列表"""
    face_db = load_face_db()
    return list(face_db.values())


def delete_face(name: str) -> Dict:
    """
    删除户主
    
    Args:
        name: 户主姓名
    
    Returns:
        删除结果 {success, message}
    """
    face_db = load_face_db()
    
    if name not in face_db:
        return {
            'success': False,
            'message': f'户主 {name} 不存在'
        }
    
    # 删除特征文件
    embedding_file = os.path.join(DEFAULT_EMBEDDINGS_DIR, f'{name}.npy')
    if os.path.exists(embedding_file):
        os.remove(embedding_file)
    
    # 从数据库删除
    del face_db[name]
    save_face_db(face_db)
    
    return {
        'success': True,
        'message': f'户主 {name} 已删除'
    }


def load_all_embeddings() -> Dict[str, np.ndarray]:
    """
    加载所有户主的特征向量
    
    Returns:
        {name: embedding}
    """
    ensure_face_db_dirs()
    embeddings = {}
    
    face_db = load_face_db()
    for name in face_db.keys():
        embedding_file = os.path.join(DEFAULT_EMBEDDINGS_DIR, f'{name}.npy')
        if os.path.exists(embedding_file):
            try:
                embeddings[name] = np.load(embedding_file)
            except Exception as e:
                logger.error(f"加载 {name} 的特征向量失败: {e}")
    
    return embeddings


# ==================== 视频人脸识别 ====================

def recognize_face(embedding: np.ndarray, threshold: float = 0.5) -> Tuple[str, float]:
    """
    识别人脸
    
    Args:
        embedding: 待识别的人脸特征向量
        threshold: 相似度阈值
    
    Returns:
        (name, similarity) 如果未识别则 name 为 "未知"
    """
    all_embeddings = load_all_embeddings()
    
    if not all_embeddings:
        return "未知", 0.0
    
    best_match = "未知"
    best_similarity = 0.0
    
    for name, db_embedding in all_embeddings.items():
        similarity = compare_embeddings(embedding, db_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            if similarity >= threshold:
                best_match = name
    
    return best_match, best_similarity


def extract_faces_from_video(video_path: str, fps: float = 1.0, max_frames: int = 30) -> List[Dict]:
    """
    从视频中提取人脸并识别
    
    Args:
        video_path: 视频文件路径
        fps: 提取帧率
        max_frames: 最大帧数
    
    Returns:
        识别结果列表 [{frame, time, faces: [{name, similarity, bbox}]}]
    """
    try:
        import cv2
    except ImportError:
        logger.error("需要安装 opencv-python")
        return []
    
    fa = get_face_analysis()
    if fa is None:
        logger.error("InsightFace 模型未初始化")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)
    frame_interval = max(1, frame_interval)
    
    results = []
    frame_index = 0
    extracted_count = 0
    
    while extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_interval == 0:
            # 当前时间（秒）
            current_time = frame_index / video_fps if video_fps > 0 else 0
            
            # 检测人脸
            faces = fa.get(frame)
            
            frame_result = {
                'frame': extracted_count + 1,
                'frame_index': frame_index,
                'time': round(current_time, 2),
                'faces': []
            }
            
            for face in faces:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    # 识别人脸
                    name, similarity = recognize_face(face.embedding)
                    
                    face_info = {
                        'name': name,
                        'similarity': round(similarity, 3),
                        'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
                        'det_score': round(float(face.det_score), 3) if hasattr(face, 'det_score') else 0
                    }
                    frame_result['faces'].append(face_info)
            
            results.append(frame_result)
            extracted_count += 1
        
        frame_index += 1
    
    cap.release()
    
    logger.info(f"视频分析完成，共分析 {len(results)} 帧")
    return results


def summarize_recognition_results(results: List[Dict]) -> Dict:
    """
    汇总识别结果
    
    Args:
        results: extract_faces_from_video 的返回结果
    
    Returns:
        汇总信息 {total_frames, faces_detected, recognized_persons, unknown_count, similarity_stats, details}
    """
    total_frames = len(results)
    faces_detected = 0
    person_appearances = {}  # {name: count}
    person_similarities = {}  # {name: [similarity1, similarity2, ...]}
    unknown_count = 0
    unknown_similarities = []
    
    for frame_result in results:
        for face in frame_result.get('faces', []):
            faces_detected += 1
            name = face.get('name', '未知')
            similarity = face.get('similarity', 0)
            
            if name == '未知':
                unknown_count += 1
                unknown_similarities.append(similarity)
            else:
                person_appearances[name] = person_appearances.get(name, 0) + 1
                if name not in person_similarities:
                    person_similarities[name] = []
                person_similarities[name].append(similarity)
    
    # 计算每个人的相似度统计
    similarity_stats = {}
    for name, similarities in person_similarities.items():
        if similarities:
            similarity_stats[name] = {
                'max': round(max(similarities), 3),
                'min': round(min(similarities), 3),
                'avg': round(sum(similarities) / len(similarities), 3),
                'count': len(similarities)
            }
    
    # 生成摘要文本（包含相似度信息）
    summary_parts = []
    if person_appearances:
        for name, count in sorted(person_appearances.items(), key=lambda x: -x[1]):
            stats = similarity_stats.get(name, {})
            max_sim = stats.get('max', 0)
            avg_sim = stats.get('avg', 0)
            summary_parts.append(f"{name}(出现{count}次, 最高相似度{max_sim*100:.1f}%, 平均{avg_sim*100:.1f}%)")
        summary_text = "识别到户主: " + ", ".join(summary_parts)
    else:
        summary_text = "未识别到已注册的户主"
    
    if unknown_count > 0:
        summary_text += f"；检测到{unknown_count}次未知人脸"
    
    return {
        'total_frames': total_frames,
        'faces_detected': faces_detected,
        'recognized_persons': person_appearances,
        'similarity_stats': similarity_stats,
        'unknown_count': unknown_count,
        'summary': summary_text
    }


# ==================== 图像标注功能 ====================

def get_chinese_font_path() -> Optional[str]:
    """
    自动检测系统中文字体路径
    
    优先级：
    1. 项目目录下的自定义字体 (font.ttf / font.ttc)
    2. 系统中文字体
    
    Returns:
        字体文件路径，如果未找到则返回 None
    """
    import platform
    system = platform.system()
    
    # 首先检查项目目录下是否有自定义字体
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_fonts = [
        os.path.join(script_dir, 'font.ttf'),
        os.path.join(script_dir, 'font.ttc'),
        os.path.join(script_dir, 'chinese_font.ttf'),
        os.path.join(script_dir, 'chinese_font.ttc'),
        os.path.join(script_dir, 'msyh.ttc'),
        os.path.join(script_dir, 'simhei.ttf'),
    ]
    
    for custom_font in custom_fonts:
        if os.path.exists(custom_font):
            logger.info(f"使用项目目录下的自定义字体: {custom_font}")
            return custom_font
    
    # 系统字体路径
    font_paths = []
    
    if system == 'Windows':
        # Windows 字体目录
        import os as os_module
        windows_font_dir = os_module.path.join(os_module.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        
        font_names = [
            'msyh.ttc',      # 微软雅黑
            'msyhbd.ttc',    # 微软雅黑粗体  
            'simhei.ttf',    # 黑体
            'simsun.ttc',    # 宋体
            'simkai.ttf',    # 楷体
            'STKAITI.TTF',   # 华文楷体
            'STSONG.TTF',    # 华文宋体
            'STXIHEI.TTF',   # 华文细黑
            'msyhl.ttc',     # 微软雅黑 Light
            'Deng.ttf',      # 等线
            'Dengb.ttf',     # 等线粗体
        ]
        
        for font_name in font_names:
            # 检查原始文件名
            font_paths.append(os_module.path.join(windows_font_dir, font_name))
            # 检查小写
            font_paths.append(os_module.path.join(windows_font_dir, font_name.lower()))
            # 检查大写
            font_paths.append(os_module.path.join(windows_font_dir, font_name.upper()))
            
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        ]
    
    for path in font_paths:
        if os.path.exists(path):
            logger.info(f"找到中文字体: {path}")
            return path
    
    logger.warning("未找到中文字体，标签将使用英文显示")
    logger.warning(f"提示：您可以将中文字体文件复制到项目目录 ({script_dir}) 并命名为 font.ttf 或 font.ttc")
    return None


# 缓存字体路径
_chinese_font_path = None
_font_checked = False


def list_available_fonts() -> List[str]:
    """列出系统中可用的字体文件（用于调试）"""
    import platform
    import glob
    
    system = platform.system()
    found_fonts = []
    
    if system == 'Windows':
        import os as os_module
        windows_font_dir = os_module.path.join(os_module.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        patterns = ['*.ttf', '*.ttc', '*.otf', '*.TTF', '*.TTC', '*.OTF']
        for pattern in patterns:
            found_fonts.extend(glob.glob(os_module.path.join(windows_font_dir, pattern)))
    elif system == 'Darwin':
        font_dirs = ['/System/Library/Fonts', '/Library/Fonts', '~/Library/Fonts']
        for font_dir in font_dirs:
            font_dir = os.path.expanduser(font_dir)
            if os.path.exists(font_dir):
                found_fonts.extend(glob.glob(os.path.join(font_dir, '*.ttf')))
                found_fonts.extend(glob.glob(os.path.join(font_dir, '*.ttc')))
    else:
        font_dirs = ['/usr/share/fonts', '/usr/local/share/fonts']
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for f in files:
                        if f.endswith(('.ttf', '.ttc', '.otf')):
                            found_fonts.append(os.path.join(root, f))
    
    return found_fonts[:50]  # 只返回前50个


def get_cached_font_path() -> Optional[str]:
    """获取缓存的字体路径"""
    global _chinese_font_path, _font_checked
    if not _font_checked:
        _chinese_font_path = get_chinese_font_path()
        if _chinese_font_path is None:
            # 输出可用字体列表帮助调试
            available = list_available_fonts()
            if available:
                logger.info(f"系统可用字体 (前10个): {available[:10]}")
        _font_checked = True
    return _chinese_font_path


def draw_face_boxes(frame, faces: List[Dict], threshold: float = 0.7) -> np.ndarray:
    """
    在帧图片上绘制人脸边框和标签
    
    Args:
        frame: OpenCV 格式的帧图片 (BGR)
        faces: 人脸列表 [{name, similarity, bbox, ...}]
        threshold: 相似度阈值，>=threshold 为户主，<threshold 为陌生人
    
    Returns:
        标注后的帧图片 (BGR)
    """
    from PIL import Image, ImageDraw, ImageFont
    
    if not faces:
        return frame
    
    # BGR 转 RGB
    frame_rgb = frame[:, :, ::-1].copy()
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # 加载字体
    font_path = get_cached_font_path()
    font_size = 24
    font = None
    use_chinese = False
    
    # 尝试加载中文字体
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            # 测试是否能正确渲染中文
            test_bbox = draw.textbbox((0, 0), "测试", font=font)
            if test_bbox[2] - test_bbox[0] > 0:
                use_chinese = True
                logger.info(f"成功加载中文字体: {font_path}")
        except Exception as e:
            logger.warning(f"加载中文字体失败: {e}")
            font = None
    
    # 如果中文字体加载失败，使用默认字体
    if font is None:
        try:
            # 尝试使用 Pillow 内置的字体
            font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"加载默认字体失败: {e}")
    
    for face in faces:
        bbox = face.get('bbox')
        if not bbox:
            continue
        
        name = face.get('name', '未知')
        similarity = face.get('similarity', 0)
        
        # 坐标
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # 判断是户主还是陌生人
        if name != '未知' and similarity >= threshold:
            # 户主 - 绿色
            box_color = (0, 200, 0)  # RGB 绿色
            if use_chinese:
                label = f"{name} {similarity*100:.0f}%"
            else:
                # 英文 fallback
                label = f"{name} {similarity*100:.0f}%"
        else:
            # 陌生人 - 红色
            box_color = (220, 50, 50)  # RGB 红色
            if use_chinese:
                label = "陌生人"
            else:
                # 英文 fallback
                label = "Stranger"
        
        # 绘制边框
        box_width = 3
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
        
        # 计算标签背景大小
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except (AttributeError, TypeError):
            # 旧版 Pillow 或无字体
            text_width = len(label) * 12
            text_height = 20
        
        # 确保最小尺寸
        text_width = max(text_width, 20)
        text_height = max(text_height, 16)
        
        # 标签背景位置（框上方）
        label_x = x1
        label_y = y1 - text_height - 10
        if label_y < 0:
            label_y = y2 + 4  # 如果上方空间不够，放到下方
        
        # 绘制标签背景
        padding = 4
        draw.rectangle(
            [label_x - padding, label_y - padding, 
             label_x + text_width + padding, label_y + text_height + padding],
            fill=box_color
        )
        
        # 绘制标签文字（白色）
        try:
            draw.text((label_x, label_y), label, font=font, fill=(255, 255, 255))
        except Exception as e:
            logger.warning(f"绘制文字失败: {e}")
    
    # RGB 转回 BGR
    result = np.array(pil_img)[:, :, ::-1]
    return result


def extract_faces_from_video_with_annotations(
    video_path: str, 
    fps: float = 1.0, 
    max_frames: int = 30,
    threshold: float = 0.7
) -> Tuple[List[Dict], List[Dict]]:
    """
    从视频中提取人脸并识别，同时生成标注后的帧图片
    
    Args:
        video_path: 视频文件路径
        fps: 提取帧率
        max_frames: 最大帧数
        threshold: 相似度阈值，>=threshold 为户主，<threshold 为陌生人
    
    Returns:
        (识别结果列表, 标注帧列表)
        标注帧列表: [{frame, time, image_base64, faces_count, recognized_count, stranger_count}]
    """
    import cv2
    import base64
    
    fa = get_face_analysis()
    if fa is None:
        logger.error("InsightFace 模型未初始化")
        return [], []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return [], []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)
    frame_interval = max(1, frame_interval)
    
    results = []
    annotated_frames = []
    frame_index = 0
    extracted_count = 0
    
    while extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % frame_interval == 0:
            current_time = frame_index / video_fps if video_fps > 0 else 0
            
            # 检测人脸
            faces = fa.get(frame)
            
            frame_result = {
                'frame': extracted_count + 1,
                'frame_index': frame_index,
                'time': round(current_time, 2),
                'faces': []
            }
            
            face_infos = []
            recognized_count = 0
            stranger_count = 0
            
            for face in faces:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    name, similarity = recognize_face(face.embedding)
                    
                    face_info = {
                        'name': name,
                        'similarity': round(similarity, 3),
                        'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
                        'det_score': round(float(face.det_score), 3) if hasattr(face, 'det_score') else 0
                    }
                    frame_result['faces'].append(face_info)
                    face_infos.append(face_info)
                    
                    # 统计
                    if name != '未知' and similarity >= threshold:
                        recognized_count += 1
                    else:
                        stranger_count += 1
            
            results.append(frame_result)
            
            # 只有检测到人脸的帧才生成标注图片
            if face_infos:
                # 绘制标注
                annotated_frame = draw_face_boxes(frame, face_infos, threshold)
                
                # 编码为 base64
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                annotated_frames.append({
                    'frame': extracted_count + 1,
                    'time': round(current_time, 2),
                    'image_base64': image_base64,
                    'faces_count': len(face_infos),
                    'recognized_count': recognized_count,
                    'stranger_count': stranger_count
                })
            
            extracted_count += 1
        
        frame_index += 1
    
    cap.release()
    
    logger.info(f"视频分析完成，共分析 {len(results)} 帧，生成 {len(annotated_frames)} 张标注图")
    return results, annotated_frames


# ==================== 视频标注功能 ====================

def draw_labels_on_frame(frame, faces: List[Dict], threshold: float = 0.7, show_unknown: bool = True) -> np.ndarray:
    """
    在帧上绘制英文标签（使用 OpenCV，无需中文字体）
    
    Args:
        frame: OpenCV 格式的帧图片 (BGR)
        faces: 人脸列表 [{name, similarity, bbox}]
        threshold: 相似度阈值
        show_unknown: 是否显示未知人脸
    
    Returns:
        标注后的帧图片
    """
    import cv2
    
    result = frame.copy()
    
    for face in faces:
        bbox = face.get('bbox')
        if not bbox:
            continue
        
        name = face.get('name', 'Unknown')
        similarity = face.get('similarity', 0)
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # 判断是否识别成功
        if name != 'Unknown' and similarity >= threshold:
            # 识别成功 - 绿色
            color = (0, 200, 0)  # BGR
            label = f"{name} {similarity*100:.0f}%"
        else:
            # 未识别 - 黄色
            if not show_unknown:
                continue
            color = (0, 200, 255)  # BGR 黄色
            label = "Unknown"
        
        # 绘制边框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # 计算标签位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 标签背景位置
        label_y = y1 - 10
        if label_y - text_height < 0:
            label_y = y2 + text_height + 10
        
        # 绘制标签背景
        cv2.rectangle(result, 
                      (x1, label_y - text_height - 5), 
                      (x1 + text_width + 5, label_y + 5), 
                      color, -1)
        
        # 绘制标签文字（白色）
        cv2.putText(result, label, (x1 + 2, label_y), font, font_scale, (255, 255, 255), thickness)
    
    return result


def label_video_with_faces(
    video_path: str,
    person_embeddings: Dict[str, np.ndarray],
    output_path: str,
    threshold: float = 0.7,
    show_unknown: bool = True,
    progress_callback=None
) -> Dict:
    """
    对视频进行人脸标注并输出标注后的视频
    
    Args:
        video_path: 输入视频路径
        person_embeddings: 人物特征向量字典 {name: embedding}
        output_path: 输出视频路径
        threshold: 相似度阈值
        show_unknown: 是否显示未知人脸
        progress_callback: 进度回调函数 callback(current, total)
    
    Returns:
        处理结果 {success, message, output_path, stats}
    """
    import cv2
    
    fa = get_face_analysis()
    if fa is None:
        return {'success': False, 'message': 'InsightFace 模型未初始化'}
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'success': False, 'message': f'无法打开视频: {video_path}'}
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        return {'success': False, 'message': '无法创建输出视频文件'}
    
    # 统计信息
    stats = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'faces_detected': 0,
        'faces_recognized': 0,
        'person_counts': {}
    }
    
    frame_index = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测人脸
            faces = fa.get(frame)
            
            face_infos = []
            for face in faces:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    # 与所有人物比对
                    best_name = 'Unknown'
                    best_similarity = 0.0
                    
                    for name, emb in person_embeddings.items():
                        sim = compare_embeddings(face.embedding, emb)
                        if sim > best_similarity:
                            best_similarity = sim
                            if sim >= threshold:
                                best_name = name
                    
                    face_info = {
                        'name': best_name,
                        'similarity': best_similarity,
                        'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None
                    }
                    face_infos.append(face_info)
                    
                    # 统计
                    stats['faces_detected'] += 1
                    if best_name != 'Unknown':
                        stats['faces_recognized'] += 1
                        stats['person_counts'][best_name] = stats['person_counts'].get(best_name, 0) + 1
            
            # 绘制标签
            labeled_frame = draw_labels_on_frame(frame, face_infos, threshold, show_unknown)
            
            # 写入帧
            out.write(labeled_frame)
            
            frame_index += 1
            stats['processed_frames'] = frame_index
            
            # 进度回调
            if progress_callback and frame_index % 10 == 0:
                progress_callback(frame_index, total_frames)
        
        cap.release()
        out.release()
        
        logger.info(f"视频标注完成: {output_path}")
        
        return {
            'success': True,
            'message': '视频标注完成',
            'output_path': output_path,
            'stats': stats
        }
        
    except Exception as e:
        cap.release()
        out.release()
        logger.error(f"视频标注失败: {e}")
        return {'success': False, 'message': str(e)}


# ==================== 实时标注功能 ====================

def realtime_label_generator(
    video_path: str,
    person_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.7,
    show_unknown: bool = True,
    target_fps: float = 2.0,
    collect_appearances: bool = False
):
    """
    实时标注生成器，逐帧 yield 标注后的图片
    
    Args:
        video_path: 视频路径
        person_embeddings: 人物特征向量字典 {name: embedding}
        threshold: 相似度阈值
        show_unknown: 是否显示未知人脸
        target_fps: 目标处理帧率（降低以提高速度）
        collect_appearances: 是否收集 owner 出场帧用于场景分析
    
    Yields:
        dict: {frame_index, total_frames, image_base64, faces, fps, progress}
    """
    import cv2
    import base64
    
    fa = get_face_analysis()
    if fa is None:
        yield {'error': 'InsightFace 模型未初始化'}
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield {'error': f'无法打开视频: {video_path}'}
        return
    
    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算跳帧间隔
    frame_interval = max(1, int(video_fps / target_fps))
    
    logger.info(f"实时标注开始: {total_frames} 帧, 视频FPS={video_fps}, 处理间隔={frame_interval}")
    
    # 发送视频信息
    yield {
        'type': 'info',
        'total_frames': total_frames,
        'video_fps': video_fps,
        'width': width,
        'height': height,
        'frame_interval': frame_interval
    }
    
    frame_index = 0
    processed_count = 0
    
    # Owner 出场帧记录 (用于场景分析)
    owner_appearances = {}  # {name: [{'frame_index': int, 'time': str, 'time_seconds': float, 'image_base64': str, 'similarity': float}]}
    owner_last_record_time = {}  # 记录每个 owner 最后一次被记录的时间（秒），用于采样间隔控制
    MIN_RECORD_INTERVAL = 2.0  # 同一个 owner 至少间隔 2 秒才记录
    MAX_RECORDS_PER_OWNER = 10  # 每人最多记录 10 帧
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳帧处理
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue
            
            # 检测人脸
            faces = fa.get(frame)
            
            face_infos = []
            current_time_seconds = frame_index / video_fps
            
            for face in faces:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    # 与所有人物比对
                    best_name = 'Unknown'
                    best_similarity = 0.0
                    
                    for name, emb in person_embeddings.items():
                        sim = compare_embeddings(face.embedding, emb)
                        if sim > best_similarity:
                            best_similarity = sim
                            if sim >= threshold:
                                best_name = name
                    
                    face_info = {
                        'name': best_name,
                        'similarity': best_similarity,
                        'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None
                    }
                    face_infos.append(face_info)
                    
                    # 收集 owner 出场帧 (非 Unknown 且启用收集)
                    if collect_appearances and best_name != 'Unknown':
                        last_time = owner_last_record_time.get(best_name, -MIN_RECORD_INTERVAL)
                        current_records = owner_appearances.get(best_name, [])
                        
                        # 检查采样条件：间隔足够且未超过最大记录数
                        if (current_time_seconds - last_time >= MIN_RECORD_INTERVAL and 
                            len(current_records) < MAX_RECORDS_PER_OWNER):
                            
                            # 将原始帧编码为 base64 (用于 Gemini 分析)
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                            _, raw_buffer = cv2.imencode('.jpg', frame, encode_param)
                            raw_base64 = base64.b64encode(raw_buffer).decode('utf-8')
                            
                            # 格式化时间
                            minutes = int(current_time_seconds // 60)
                            seconds = int(current_time_seconds % 60)
                            time_str = f"{minutes:02d}:{seconds:02d}"
                            
                            appearance_record = {
                                'frame_index': frame_index,
                                'time': time_str,
                                'time_seconds': current_time_seconds,
                                'image_base64': raw_base64,
                                'similarity': round(best_similarity, 2)
                            }
                            
                            if best_name not in owner_appearances:
                                owner_appearances[best_name] = []
                            owner_appearances[best_name].append(appearance_record)
                            owner_last_record_time[best_name] = current_time_seconds
                            
                            logger.debug(f"记录 {best_name} 出场帧: {time_str}, 相似度: {best_similarity:.2f}")
            
            # 绘制标签
            labeled_frame = draw_labels_on_frame(frame, face_infos, threshold, show_unknown)
            
            # 压缩图片质量以加快传输
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', labeled_frame, encode_param)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            processed_count += 1
            progress = round(frame_index / total_frames * 100, 1)
            current_time = round(frame_index / video_fps, 2)
            
            yield {
                'type': 'frame',
                'frame_index': frame_index,
                'total_frames': total_frames,
                'current_time': current_time,
                'image_base64': image_base64,
                'faces': face_infos,
                'progress': progress
            }
            
            frame_index += 1
        
        # 发送完成信号（包含 owner 出场帧数据）
        complete_data = {
            'type': 'complete',
            'processed_count': processed_count,
            'total_frames': total_frames
        }
        
        if collect_appearances and owner_appearances:
            complete_data['owner_appearances'] = owner_appearances
            logger.info(f"收集到 {len(owner_appearances)} 个 owner 的出场帧")
        
        yield complete_data
        
    except Exception as e:
        logger.error(f"实时标注错误: {e}")
        yield {'type': 'error', 'error': str(e)}
    finally:
        cap.release()
        logger.info(f"实时标注结束，共处理 {processed_count} 帧")

