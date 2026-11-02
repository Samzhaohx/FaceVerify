# -*- coding: utf-8 -*-
"""静默活体检测（MiniFASNetV2，单帧频域纹理分析）。

思路：真人皮肤在成像链路（镜头 + 屏幕/纸张材质）中会留下一致的
高频纹理与反射特征；打印件、屏幕翻拍、重放视频在频域上表现为
纹理塌陷或摩尔纹。MiniFASNetV2 直接在裁剪后的单帧上做 3 分类。

工程要点（都是实测踩出来的）：
  · 输入为 BGR、0~255 原始像素，**不做** mean/std 归一化，模型内部有 BN
  · 裁剪时不能只贴着脸裁，必须外扩 2.7 倍带上耳侧与额头发际，否则真人也判假
  · 单模型 + 单裁剪方差大，这里做 多尺度 + 水平翻转 的多视图集成
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .ort_utils import create_session

log = logging.getLogger(__name__)


class LivenessDetector:
    # 官方 3 分类输出顺序为 [fake, real, fake]（索引 1 = 真人）
    REAL_CLASS_INDEX = 1

    def __init__(
        self,
        model_path: str,
        prefer_gpu: bool = True,
        scales: tuple[float, ...] = (2.7,),
        use_flip: bool = True,
    ):
        self.session = create_session(model_path, prefer_gpu=prefer_gpu, name="MiniFASNetV2")
        in0 = self.session.get_inputs()[0]
        self.input_name = in0.name
        shape = in0.shape
        self.input_size = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 80
        self.scales = tuple(scales)
        self.use_flip = use_flip

    # ---------------- 裁剪 ----------------

    @staticmethod
    def _expand_square(bbox: np.ndarray, scale: float) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * scale
        return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)

    @staticmethod
    def _crop(img: np.ndarray, box: np.ndarray, size: int) -> np.ndarray | None:
        """把人脸框外扩成正方形后裁出来。

        关键细节：框越界时**不能补黑边**。实测同一个真人，裁剪框里只要出现
        两条黑边（例如竖屏素材被 letterbox 到横屏），MiniFASNetV2 的真人分数
        会从 1.000 直接掉到 0.003 —— 它把黑色边界当成了屏幕/打印件的边缘。
        正确做法是把框的中心拉回画面内，必要时缩小边长，保证裁剪全部落在图像里。
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        side = max(x2 - x1, y2 - y1)
        side = min(side, float(min(w, h)))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = side / 2.0
        if side < w:
            cx = min(max(cx, half), w - half)
        else:
            cx = w / 2.0
        if side < h:
            cy = min(max(cy, half), h - half)
        else:
            cy = h / 2.0

        xi1 = int(round(max(0.0, cx - half)))
        yi1 = int(round(max(0.0, cy - half)))
        xi2 = int(round(min(float(w), cx + half)))
        yi2 = int(round(min(float(h), cy + half)))
        if xi2 - xi1 < 20 or yi2 - yi1 < 20:
            return None
        patch = img[yi1:yi2, xi1:xi2]
        return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)

    # ---------------- 推理 ----------------

    def _forward(self, patches: list[np.ndarray]) -> np.ndarray:
        blob = np.stack([np.transpose(p.astype(np.float32), (2, 0, 1)) for p in patches])
        out = self.session.run(None, {self.input_name: blob})[0]
        # 稳定 softmax
        out = out - out.max(axis=1, keepdims=True)
        e = np.exp(out)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, img_bgr: np.ndarray, face_box: np.ndarray) -> tuple[float, dict]:
        """返回 (真人置信度, 诊断信息)。face_box 建议传 106 点外接框。"""
        probs_all = []
        debug: dict = {}
        patches = []

        for scale in self.scales:
            box = self._expand_square(face_box, scale)
            patch = self._crop(img_bgr, box, self.input_size)
            if patch is None:
                continue
            patches.append(patch)
            views = [patch]
            if self.use_flip:
                views.append(cv2.flip(patch, 1))
            probs = self._forward(views)          # (2, 3)
            probs_all.append(probs)
            if len(self.scales) > 1:
                debug[f"scale{scale}"] = round(float(probs[:, self.REAL_CLASS_INDEX].mean()), 3)
        if not probs_all:
            return 0.0, {"error": "crop_failed"}

        probs = np.concatenate(probs_all, axis=0)   # (n_view, 3)
        live = float(probs[:, self.REAL_CLASS_INDEX].mean())
        # 融合方差：视图之间分歧越大说明越可疑，向 0.5 收缩
        spread = float(probs[:, self.REAL_CLASS_INDEX].std())
        if spread > 0.15:
            live = 0.5 + (live - 0.5) * 0.85

        # 画质指标：在送入模型的同一张 patch 上算，才是模型"看到"的东西
        patch = patches[0]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        debug.update(
            spread=round(spread, 3),
            views=int(probs.shape[0]),
            brightness=round(float(gray.mean()), 1),
            sharpness=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            face_px=int(round(max(face_box[2] - face_box[0], face_box[3] - face_box[1]))),
            dark_edge=round(_dark_edge_ratio(gray), 3),
        )
        return float(np.clip(live, 0.0, 1.0)), debug


def _dark_edge_ratio(gray_patch: np.ndarray, ring: int = 8, level: int = 32) -> float:
    """裁剪图四周一圈里"接近全黑"的像素占比。

    用来发现画面自带的黑边（letterbox、播放器边框、虚拟背景）。
    实测这类黑边会让真人活体分数从 1.000 掉到 0.003，是误判的头号原因，
    所以值得单独量化。注意不能用"裁剪框是否越界"判断——2.7 倍外扩几乎必然越界，
    对所有人都成立，没有区分度。
    """
    h, w = gray_patch.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[ring:h - ring, ring:w - ring] = False
    total = int(mask.sum())
    if total == 0:
        return 0.0
    return float((gray_patch[mask] < level).sum() / total)


# 阈值按 80x80 patch 实测标定（见 docs/DEVNOTES.md 第 15 条）
DARK_LEVEL = 55.0        # 平均灰度低于此值算偏暗
BRIGHT_LEVEL = 205.0     # 高于此值算过曝
BLUR_LEVEL = 12.0        # Laplacian 方差低于此值算模糊（80x80 尺度）
SMALL_FACE_PX = 110      # 人脸宽度低于此值，纹理信息不足
DARK_EDGE_LEVEL = 0.35   # 四周黑像素占比超过此值，认为画面带黑边


def diagnose(live: float, debug: dict, threshold: float, face_px: int | None = None) -> list[str]:
    """把"活体不过"翻译成人能看懂的几条原因。

    排序原则：**先给具体、可操作的原因**（黑边/太小/太暗/模糊），
    再给分数层面的结论。反过来会把"画面有黑边"这种能立刻修的问题
    埋在"分数偏低"这种废话后面。
    """
    if debug.get("error"):
        return ["人脸裁剪失败，请正对镜头"]

    px = face_px or debug.get("face_px") or 0
    brightness = debug.get("brightness", 128)
    sharpness = debug.get("sharpness", 999)
    spread = debug.get("spread", 0)
    dark_edge = debug.get("dark_edge", 0)

    causes: list[str] = []
    # ---- 具体画质原因 ----
    # 黑边只在"人脸曝光正常、四周却发黑"时才成立，整体偏暗的房间要先排除
    if dark_edge >= DARK_EDGE_LEVEL and brightness >= DARK_LEVEL:
        causes.append(f"画面四周有黑边（{dark_edge*100:.0f}%），易被误判为屏幕边框")
    if px and px < SMALL_FACE_PX:
        causes.append(f"人脸偏小（{px}px），请靠近一点")
    if brightness < DARK_LEVEL:
        causes.append(f"画面偏暗（{brightness:.0f}），请正对光源")
    elif brightness > BRIGHT_LEVEL:
        causes.append(f"画面过曝（{brightness:.0f}），避免强光直射")
    if sharpness < BLUR_LEVEL:
        causes.append(f"画面模糊（清晰度 {sharpness:.0f}），擦拭镜头")
    if spread > 0.15:
        causes.append(f"多视图判断不一致（{spread:.2f}），可能有遮挡")

    # ---- 结论 ----
    if not causes:
        if live < 0.15:
            causes.append("纹理特征强烈异常，判定为照片/屏幕翻拍")
        elif live < 0.35:
            causes.append("纹理特征偏向翻拍，疑似照片或屏幕")
        else:
            causes.append(f"活体 {live:.2f} 略低于阈值 {threshold:.2f}，调整光照或角度后可重试")
    elif live < 0.15:
        causes.append("纹理特征同时强烈异常，建议按攻击处理")

    return causes[:3]
