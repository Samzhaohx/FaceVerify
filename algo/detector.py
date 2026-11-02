# -*- coding: utf-8 -*-
"""SCRFD 人脸检测（InsightFace buffalo_s / det_500m.onnx）。

相对原项目的 Haar 级联：
  · Haar 只能正脸，侧脸/遮挡直接漏检；SCRFD 是 anchor-based CNN，多尺度、能出 5 点关键点
  · 一次前向同时得到 人脸框 + 置信度 + 5 关键点，省掉单独的 dlib 68 点预测
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .ort_utils import create_session

log = logging.getLogger(__name__)


@dataclass
class Face:
    """一次检测结果。bbox / kps 均为原图坐标系。"""

    bbox: np.ndarray            # [x1, y1, x2, y2]
    score: float                # 检测置信度
    kps: np.ndarray             # (5, 2) 左右眼、鼻尖、左右嘴角
    kps106: np.ndarray | None = None   # (106, 2) 精修关键点，可能为 None
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (float((self.bbox[0] + self.bbox[2]) / 2), float((self.bbox[1] + self.bbox[3]) / 2))


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray, max_shape: tuple[int, int]) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        px = np.clip(px, 0, max_shape[1] - 1)
        py = np.clip(py, 0, max_shape[0] - 1)
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    """标准 IoU NMS，dets 为 [x1,y1,x2,y2,score] 的 ndarray。"""
    if len(dets) == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


class SCRFDDetector:
    """SCRFD 检测器封装。strides/anchor 数按 buffalo_s 的 det_500m 配置。"""

    def __init__(
        self,
        model_path: str,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        prefer_gpu: bool = True,
    ):
        self.session = create_session(model_path, prefer_gpu=prefer_gpu, name="SCRFD-det")
        self.input_mean = 127.5
        self.input_std = 128.0
        self.det_size = tuple(det_size)
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh

        in0 = self.session.get_inputs()[0]
        self.input_name = in0.name
        # 动态输入（None/?）时用配置的 det_size；固定输入时跟随模型
        shape = in0.shape
        if isinstance(shape[2], int) and shape[2] > 0:
            self.det_size = (int(shape[2]), int(shape[3]))

        self._fmc = 3
        self._strides = [8, 16, 32]
        self._num_anchors = 2
        self._anchor_cache: dict[tuple[int, int], np.ndarray] = {}

    def _anchors(self, height: int, width: int, stride: int) -> np.ndarray:
        """height/width 是**已经除过 stride 的特征图尺寸**，不要重复除。"""
        key = (height, width, stride)
        if key in self._anchor_cache:
            return self._anchor_cache[key]
        ac = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        ac = (ac * stride).reshape((-1, 2))
        if self._num_anchors > 1:
            ac = np.stack([ac] * self._num_anchors, axis=1).reshape((-1, 2))
        self._anchor_cache[key] = ac
        return ac

    def detect(self, img: np.ndarray) -> list[Face]:
        """img: BGR uint8。返回按面积降序的人脸列表。"""
        im_ratio = float(img.shape[0]) / img.shape[1]
        det_w, det_h = self.det_size
        model_ratio = float(det_h) / det_w

        if im_ratio > model_ratio:
            new_h = det_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = det_w
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / img.shape[0]

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((det_h, det_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / self.input_std, (det_w, det_h), (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        net_outs = self.session.run(None, {self.input_name: blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        fmc = self._fmc
        for idx, stride in enumerate(self._strides):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride

            height = det_h // stride
            width = det_w // stride
            anchor_centers = self._anchors(height, width, stride)
            scores = scores.reshape(-1)

            pos_inds = np.where(scores >= self.det_thresh)[0]
            if pos_inds.size == 0:
                continue
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            kpss = _distance2kps(anchor_centers, kps_preds, max_shape=canvas.shape[:2])
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

        if not scores_list:
            return []

        # 注意：scores 是 1-D，必须用 concatenate；vstack 会按"行"堆叠，
        # 各 stride 的命中数不同时形状对不上
        scores = np.concatenate(scores_list).reshape(-1)
        bboxes = np.concatenate(bboxes_list, axis=0) / det_scale
        kpss = np.concatenate(kpss_list, axis=0) / det_scale

        order = scores.argsort()[::-1]
        dets = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
        keep = nms(dets, self.nms_thresh)

        faces: list[Face] = []
        for i in keep:
            b = bboxes[i].copy()
            b[0:4:2] = np.clip(b[0:4:2], 0, img.shape[1] - 1)
            b[1:4:2] = np.clip(b[1:4:2], 0, img.shape[0] - 1)
            faces.append(
                Face(
                    bbox=b.astype(np.float32),
                    score=float(scores[i]),
                    # 模型输出是 (N, 10) 的扁平关键点，这里还原成 (5, 2)
                    kps=kpss[i].reshape(5, 2).astype(np.float32),
                )
            )

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces
