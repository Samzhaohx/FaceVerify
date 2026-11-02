# -*- coding: utf-8 -*-
"""ArcFace 特征提取与人脸库比对（InsightFace buffalo_s / w600k_mbf.onnx）。

相对原项目的 dlib ResNet-34（128 维）：
  · w600k_mbf 输出 512 维 L2 归一化特征，余弦相似度区分度更高
  · 训练集为 WebFace600K（含大量亚洲人脸与遮挡、低质样本），鲁棒性更好
  · ONNX Runtime 单次推理 3~8ms（CPU），dlib 同精度下要 80ms 以上
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .align import norm_crop
from .ort_utils import create_session

log = logging.getLogger(__name__)

GALLERY_VERSION = 2


@dataclass
class MatchResult:
    name: str | None       # 命中姓名；None 表示库内无匹配
    score: float           # 余弦相似度（-1~1）
    second_name: str | None = None
    second_score: float = -1.0

    @property
    def margin(self) -> float:
        """第一名与第二名的差距，用于判断是否"撞脸"。"""
        return self.score - self.second_score


class ArcFaceRecognizer:
    INPUT_SIZE = 112

    def __init__(self, model_path: str, prefer_gpu: bool = True):
        self.session = create_session(model_path, prefer_gpu=prefer_gpu, name="ArcFace")
        in0 = self.session.get_inputs()[0]
        self.input_name = in0.name
        shape = in0.shape
        if isinstance(shape[2], int) and shape[2] > 0:
            self.INPUT_SIZE = int(shape[2])
        self.embedding_dim = int(shape[1]) if shape[1] else 512
        # 真实维度以输出为准（w600k_mbf = 512）
        out_shape = self.session.get_outputs()[0].shape
        if isinstance(out_shape[-1], int) and out_shape[-1] > 0:
            self.embedding_dim = int(out_shape[-1])

    # ---------------- 特征提取 ----------------

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        rgb = aligned_bgr[:, :, ::-1].astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        return np.transpose(rgb, (2, 0, 1))

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        blob = self._preprocess(aligned_bgr)[None]
        out = self.session.run(None, {self.input_name: blob})[0]
        v = out[0].astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_batch(self, aligned_list: list[np.ndarray]) -> np.ndarray:
        if not aligned_list:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        blob = np.stack([self._preprocess(a) for a in aligned_list])
        out = self.session.run(None, {self.input_name: blob})[0].astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def embed_from_face(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        aligned = norm_crop(img_bgr, kps, image_size=self.INPUT_SIZE)
        return self.embed(aligned)


class FaceGallery:
    """pickle 序列化的人脸特征库。支持单人多张照片（同 person 取最大相似度）。"""

    def __init__(self, path: str | Path, embeddings: np.ndarray, names: list[str], sources: list[str] | None = None):
        self.path = Path(path)
        self.embeddings = embeddings.astype(np.float32)
        self.names = list(names)
        self.sources = list(sources) if sources else [""] * len(names)
        self._centroids: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.names)

    @property
    def identities(self) -> list[str]:
        return sorted(set(self.names))

    def _centroid_map(self) -> dict[str, np.ndarray]:
        if self._centroids is None:
            m: dict[str, np.ndarray] = {}
            for name in set(self.names):
                idx = [i for i, n in enumerate(self.names) if n == name]
                c = self.embeddings[idx].mean(axis=0)
                n = np.linalg.norm(c)
                m[name] = c / n if n > 0 else c
            self._centroids = m
        return self._centroids

    def match(self, emb: np.ndarray) -> MatchResult:
        """最近邻匹配：单张照片取 max，避免平均后把姿态差异抹平。"""
        if len(self) == 0:
            return MatchResult(None, -1.0)
        sims = self.embeddings @ emb          # 全部已 L2 归一化 → 点积即余弦
        best = int(np.argmax(sims))
        best_name, best_score = self.names[best], float(sims[best])

        # 第二名（不同身份）
        others = [(float(s), n) for s, n in zip(sims, self.names) if n != best_name]
        if others:
            second_score, second_name = max(others, key=lambda x: x[0])
        else:
            second_score, second_name = -1.0, None
        return MatchResult(best_name, best_score, second_name, second_score)

    # ---------------- 持久化 ----------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GALLERY_VERSION,
            "dim": int(self.embeddings.shape[1]),
            "embeddings": self.embeddings,
            "names": self.names,
            "sources": self.sources,
        }
        with open(self.path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("人脸库已保存: %s (%d 条 / %d 人)", self.path, len(self.names), len(self.identities))

    @classmethod
    def load(cls, path: str | Path) -> "FaceGallery":
        path = Path(path)
        if not path.exists():
            log.warning("人脸库不存在: %s（将使用空库，所有人为未知）", path)
            return cls(path, np.zeros((0, 512), dtype=np.float32), [], [])
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "embeddings" in payload:
            return cls(path, payload["embeddings"], payload["names"], payload.get("sources"))
        # 兼容旧格式：{name: [emb,...]}
        names, embs, srcs = [], [], []
        for name, vectors in payload.items():
            for i, v in enumerate(vectors):
                names.append(name)
                embs.append(np.asarray(v, dtype=np.float32))
                srcs.append(f"{name}#{i}")
        return cls(path, np.vstack(embs) if embs else np.zeros((0, 512), np.float32), names, srcs)
