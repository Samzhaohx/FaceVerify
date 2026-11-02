# -*- coding: utf-8 -*-
"""InsightFace 2D106 关键点（2d106det.onnx）。

在流水线中承担两件事（都是实测验证过的）：

1. **活体输入裁剪**：用 106 点外接框代替检测框。检测框在侧脸时会把耳朵和
   大片背景带进来，而 MiniFASNet 对背景极其敏感。实测在 2.7 倍尺度下两者
   打分相当，但放大到 4.0 倍时 106 框的方差明显更小（0.996 vs 0.506）。

2. **可视化 / 姿态粗判**：把关键点画在画面上，便于调试对齐质量。

**没有**用 106 点重算 5 点做 ArcFace 对齐 —— 实测这是负优化：
同一人 5 张照片的余弦相似度均值，SCRFD 原生 5 点 0.654，106 点重映射
最好的索引组合只有 0.623，最差的 0.573。原因见 docs/DEVNOTES.md 第 4 条。
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .align import norm_crop, landmark_bbox
from .detector import Face
from .ort_utils import create_session

log = logging.getLogger(__name__)


class Landmark106:
    # 2D106 在 192x192 对齐图上推理，输出为 [-1, 1] 归一化坐标
    INPUT_SIZE = 192

    def __init__(self, model_path: str, prefer_gpu: bool = True):
        self.session = create_session(model_path, prefer_gpu=prefer_gpu, name="2D106")
        in0 = self.session.get_inputs()[0]
        self.input_name = in0.name
        shape = in0.shape
        if isinstance(shape[2], int) and shape[2] > 0:
            self.INPUT_SIZE = int(shape[2])

    def _run(self, aligned_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            aligned_bgr.astype(np.float32),
            1.0 / 128.0,
            (self.INPUT_SIZE, self.INPUT_SIZE),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        out = self.session.run(None, {self.input_name: blob})[0]
        pts = out.reshape(-1, 2).astype(np.float32)
        return (pts + 1.0) * (self.INPUT_SIZE / 2.0)

    def refine(self, img: np.ndarray, face: Face) -> Face:
        """补全 face.kps106（原图坐标）。不改动 face.kps —— 对齐仍用 SCRFD 5 点。"""
        from .align import estimate_norm

        aligned = norm_crop(img, face.kps, image_size=self.INPUT_SIZE)
        pts = self._run(aligned)
        M = estimate_norm(face.kps, self.INPUT_SIZE)
        Minv = cv2.invertAffineTransform(M)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
        face.kps106 = (pts_h @ Minv.T).astype(np.float32)
        return face

    def liveness_box(self, face: Face) -> np.ndarray:
        """给活体模型用的裁剪框：优先 106 点外接框，没有则退回检测框。"""
        if face.kps106 is not None:
            return landmark_bbox(face.kps106, 1.0)
        return face.bbox
