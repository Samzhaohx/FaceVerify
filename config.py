# -*- coding: utf-8 -*-
"""全局配置。所有可调参数集中在这里，改完重启即生效。

命名遵循原项目习惯（大写下划线），方便和入门版逐项对照。
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # ==================== 摄像头 ====================
    CAMERA_INDEX: int | str = 0        # 笔记本内置摄像头一般是 0；外接 USB 可能是 1/2
    # 采集分辨率：默认 960x540 而不是 1280x720。
    # 本机实测同一摄像头：720p 只协商到 15.4fps，960x540 有 26.3fps，640x480 有 30.3fps。
    # 540p 下 40~80cm 距离人脸仍有 200+px，识别和活体都够用，换来明显更顺的画面。
    # 需要更远的识别距离/更清晰的存档画面，再改回 1280x720（代价是帧率减半）。
    FRAME_WIDTH = 960
    FRAME_HEIGHT = 540
    CAMERA_FPS = 30                     # 期望采集帧率（摄像头实际可能不支持）
    CAMERA_BACKEND: int | None = None   # None = 自动；Windows 下优先 DSHOW
    FRAME_QUEUE_SIZE = 2                # 采集队列长度。1=只留最新帧（延迟最低），2=允许一点抖动

    # ==================== 模型 ====================
    MODEL_DIR = BASE_DIR / "models"
    DET_MODEL = "det_500m.onnx"         # SCRFD 500M，buffalo_s 套件
    LANDMARK_MODEL = "2d106det.onnx"    # 2D106 关键点
    RECOGNITION_MODEL = "w600k_mbf.onnx"  # ArcFace MobileFaceNet，512 维
    LIVENESS_MODEL = "MiniFASNetV2.onnx"  # 静默活体
    PREFER_GPU = True                   # True=优先 CUDA，失败自动降级 CPU

    # ==================== 人脸库 ====================
    DATASET_DIR = BASE_DIR / "dataset"          # 录入用的原始照片
    ENCODING_FILE = BASE_DIR / "encodings" / "faces.pkl"

    # ==================== 检测 ====================
    DET_SIZE = 640            # SCRFD 输入边长。640 足够；降到 320 可再快 2 倍但小脸会丢
    DET_THRESHOLD = 0.50      # 检测置信度阈值。误检多→调高；侧脸漏检→调到 0.35~0.45
    DET_NMS_THRESHOLD = 0.40
    MIN_FACE_SIZE = 60        # 小于该像素宽度的人脸直接丢弃（太小了活体和识别都不可靠）

    # ==================== 跳帧 / 调度 ====================
    DETECT_INTERVAL = 2       # 每 N 帧跑一次检测；中间帧靠追踪预测。低配机器可设 3~5
    RECOGNIZE_INTERVAL = 15   # 同一 track 每隔 N 帧重算一次特征，避免身份跳变
    LIVENESS_INTERVAL = 2     # 活体推理间隔（帧）

    # ==================== 风控阈值（核心调参区）====================
    # 余弦相似度：同人实测 0.55~0.75，不同人通常 < 0.30
    SIM_THRESHOLD = 0.45      # ≥ 该值判定"身份匹配"。调高更严（防误识），调低更宽松
    MISMATCH_THRESHOLD = 0.28 # 介于 MISMATCH~SIM 之间 → 身份不匹配；低于此 → 未知人员
    LIVENESS_THRESHOLD = 0.65 # 真人置信度阈值。误判真人→调低到 0.5；防照片攻击→调到 0.8
    LIVENESS_EMA = 0.5        # 活体分数指数平滑系数（0=不平滑，1=完全用历史）
    IDENTITY_CONFIRM_HITS = 3 # 连续命中多少帧后才显示姓名，抑制画面边缘的 Unknown 闪跳
    EMBEDDING_SMOOTH = 0.5    # 特征向量的新帧权重：0=不融合，1=只用当帧。0.4~0.6 可抑制相似度抖动

    # ==================== 追踪 ====================
    TRACK_IOU_THRESHOLD = 0.30
    TRACK_HIGH_THRESHOLD = 0.50   # 一级关联：高分检测
    TRACK_LOW_THRESHOLD = 0.15    # 二级关联：低分检测只用于续命
    TRACK_MAX_AGE = 30            # 连续多少帧没命中才删除轨迹
    TRACK_MIN_HITS = 3            # 连续命中多少帧才"确认"（确认前不显示，避免闪框）

    # ==================== 业务模式 ====================
    SINGLE_FACE = True        # 单人核验模式：只处理画面中面积最大的一张脸

    # ==================== 画面标注 ====================
    MIRROR = True             # 水平镜像预览（自拍视角）。只翻"显示"，不翻"分析"，见 docs/DEVNOTES.md 第 16 条
    SHOW_KEYPOINTS = False    # 是否叠加 106 点（调试用）
    SHOW_HUD = True           # 左上角信息条
    FONT_PATH: str | None = None  # 中文字体，None=自动探测（msyh.ttc / simhei.ttf ...）

    # ==================== Web 服务 ====================
    HOST = "0.0.0.0"          # 0.0.0.0 才能被局域网其他设备访问
    PORT = 5000
    JPEG_QUALITY = 80         # MJPEG 画质。60~70 更省带宽，80 足够清晰
    STREAM_FPS_LIMIT = 30     # 推流上限，避免网络带宽被吃满
    STATUS_INTERVAL_MS = 500  # 前端轮询状态间隔

    # ==================== 状态颜色（BGR）====================
    COLOR_PASSED = (80, 200, 80)      # 绿：验证通过
    COLOR_SPOOF = (60, 60, 235)       # 红：活体异常
    COLOR_MISMATCH = (40, 200, 235)   # 黄：身份不匹配
    COLOR_UNKNOWN = (235, 235, 235)   # 白：未知人员
    COLOR_HUD_BG = (28, 28, 30)
