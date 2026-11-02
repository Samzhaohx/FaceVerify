# -*- coding: utf-8 -*-
"""轻量 ByteTrack：IOU 两级关联 + 卡尔曼预测 + ID 生命周期。

为什么需要追踪：
  · 检测器每帧结果有 ±3px 抖动，直接把结果画框会"呼吸"；ID 稳定后框才稳
  · 跳帧检测时（每 N 帧才跑一次检测），中间帧靠卡尔曼预测维持框的位置
  · 识别结果与活体分数按 track_id 缓存，同一张脸不必每帧重算 ArcFace（省 80% 算力）

与原版 ByteTrack 的差异（务实取舍）：
  · 关联用"按 IOU 降序贪心"，不引入 scipy 的匈牙利算法——画面内目标数为 1~3，
    贪心与最优解几乎等价，却少一个重依赖
  · 只保留运动模型，不做 ReID 特征关联（人脸场景靠检测置信度足够）
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .detector import Face


def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4) b:(M,4) xyxy → (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(1e-6, area_a + area_b - inter)


def xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    cx = (b[..., 0] + b[..., 2]) / 2
    cy = (b[..., 1] + b[..., 3]) / 2
    w = b[..., 2] - b[..., 0]
    h = b[..., 3] - b[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x1 = b[..., 0] - b[..., 2] / 2
    y1 = b[..., 1] - b[..., 3] / 2
    return np.stack([x1, y1, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], axis=-1)


class KalmanBox:
    """匀速模型卡尔曼滤波（SORT 同款参数）。"""

    def __init__(self, bbox_xyxy: np.ndarray):
        z = xyxy_to_cxcywh(np.asarray(bbox_xyxy, dtype=np.float64))
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = z
        self.cov = np.eye(8, dtype=np.float64) * 10.0
        self.cov[4:, 4:] *= 100.0
        self._F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.zeros((4, 8), dtype=np.float64)
        self._H[:4, :4] = np.eye(4)
        self._R = np.eye(4, dtype=np.float64) * 1.0
        self._Q = np.eye(8, dtype=np.float64)
        self._Q[4:, 4:] *= 0.01

    def predict(self) -> None:
        self.mean = self._F @ self.mean
        self.cov = self._F @ self.cov @ self._F.T + self._Q

    def update(self, bbox_xyxy: np.ndarray) -> None:
        z = xyxy_to_cxcywh(np.asarray(bbox_xyxy, dtype=np.float64))
        y = z - self._H @ self.mean
        S = self._H @ self.cov @ self._H.T + self._R
        K = self.cov @ self._H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ y
        self.cov = (np.eye(8) - K @ self._H) @ self.cov

    @property
    def bbox(self) -> np.ndarray:
        b = cxcywh_to_xyxy(self.mean[:4])
        # 尺寸不允许出现负数（长时间漏检时速度外推可能翻转）
        b[[0, 2]] = np.maximum(b[[0, 2]], 0)
        b[[1, 3]] = np.maximum(b[[1, 3]], 0)
        if b[2] <= b[0]:
            b[2] = b[0] + 2
        if b[3] <= b[1]:
            b[3] = b[1] + 2
        return b.astype(np.float32)


@dataclass
class Track:
    track_id: int
    kalman: KalmanBox
    score: float = 0.0
    kps: np.ndarray | None = None
    kps106: np.ndarray | None = None
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = False
    # 业务侧缓存（身份 / 活体），由 engine 维护，tracker 不感知语义
    cache: dict = field(default_factory=dict)

    @property
    def bbox(self) -> np.ndarray:
        return self.kalman.bbox

    @property
    def center(self) -> tuple[float, float]:
        b = self.bbox
        return (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2))


class ByteTracker:
    def __init__(
        self,
        iou_thresh: float = 0.3,
        high_thresh: float = 0.5,
        low_thresh: float = 0.15,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.iou_thresh = iou_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)
        self.frame_id = 0

    # ---------------- 关联 ----------------

    def _greedy_match(self, track_idx: list[int], det_idx: list[int], iou: np.ndarray) -> tuple[list, list, list]:
        """按 IOU 降序贪心配对，返回 (matches, unmatched_tracks, unmatched_dets)。"""
        pairs = []
        for ti in track_idx:
            for di in det_idx:
                if iou[ti, di] >= self.iou_thresh:
                    pairs.append((iou[ti, di], ti, di))
        pairs.sort(key=lambda x: -x[0])
        used_t, used_d = set(), set()
        matches = []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        un_t = [t for t in track_idx if t not in used_t]
        un_d = [d for d in det_idx if d not in used_d]
        return matches, un_t, un_d

    def _apply_match(self, matches: list, dets: list[Face]) -> None:
        for ti, di in matches:
            tr = self.tracks[ti]
            det = dets[di]
            tr.kalman.update(det.bbox)
            tr.score = det.score
            tr.kps = det.kps
            tr.kps106 = det.kps106
            tr.hits += 1
            tr.time_since_update = 0
            if tr.hits >= self.min_hits:
                tr.confirmed = True
            # 命中时清掉"待重识别"标记，engine 会据此决定是否重算特征
            tr.cache["miss"] = 0

    # ---------------- 主流程 ----------------

    def update(self, dets: list[Face]) -> list[Track]:
        """喂入本帧检测结果（可空），返回当前所有活跃轨迹。"""
        self.frame_id += 1
        for tr in self.tracks:
            tr.age += 1
            tr.kalman.predict()
            tr.time_since_update += 1
            tr.cache["miss"] = tr.cache.get("miss", 0) + 1

        if not dets:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.confirmed]

        det_boxes = np.stack([d.bbox for d in dets])
        det_scores = np.array([d.score for d in dets], dtype=np.float32)
        trk_boxes = np.stack([t.bbox for t in self.tracks]) if self.tracks else np.zeros((0, 4), np.float32)
        iou = iou_batch(trk_boxes, det_boxes)

        high = [i for i in range(len(dets)) if det_scores[i] >= self.high_thresh]
        low = [i for i in range(len(dets)) if self.low_thresh <= det_scores[i] < self.high_thresh]

        all_t = list(range(len(self.tracks)))
        matches1, un_t, un_d = self._greedy_match(all_t, high, iou)
        self._apply_match(matches1, dets)

        # 第二级：低分检测只用来"续命"已有轨迹，不新建轨迹（ByteTrack 核心）
        if un_t and low:
            matches2, un_t, _ = self._greedy_match(un_t, low, iou)
            self._apply_match(matches2, dets)
            un_d = [d for d in un_d if d not in low]

        # 新轨迹：只接受高分且未被关联的检测
        for di in un_d:
            if det_scores[di] < self.high_thresh:
                continue
            tr = Track(track_id=next(self._ids), kalman=KalmanBox(dets[di].bbox), score=float(det_scores[di]))
            tr.kps = dets[di].kps
            tr.kps106 = dets[di].kps106
            tr.hits = 1
            tr.confirmed = self.min_hits <= 1
            self.tracks.append(tr)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed]

    def predict_only(self) -> list[Track]:
        """跳帧检测的空档期：只做预测，不更新观测。"""
        self.frame_id += 1
        for tr in self.tracks:
            tr.age += 1
            tr.kalman.predict()
            tr.time_since_update += 1
            tr.cache["miss"] = tr.cache.get("miss", 0) + 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed]

    def reset(self) -> None:
        self.tracks.clear()
