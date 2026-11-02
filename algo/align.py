# -*- coding: utf-8 -*-
"""人脸对齐：5 点相似变换（Umeyama）+ 106 点关键点的 5 点子集映射。

原版项目把整张图直接丢给 dlib，人脸有小角度旋转/侧脸时特征就会飘。
这里按 ArcFace 标准模板做相似变换，把眼睛、鼻尖、嘴角归一化到固定位置。
"""
import cv2
import numpy as np

# ArcFace 112x112 标准 5 点模板（insightface 同款）
ARC_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# InsightFace 2D106 关键点里对应 5 点的索引
# 0-32 脸部轮廓, 33-42 右眉, 43-51 左眉, 52-54 鼻梁, 55-59 鼻头,
# 60-63 右眼, 64-67 左眼, 68-71 上唇, 72-83 下唇, 84-103 眼部细节
KPS5_FROM_106 = {
    "right_eye": 66,  # 图像左眼（被摄者右眼）
    "left_eye": 79,   # 图像右眼（被摄者左眼）
    "nose": 54,
    "mouth_right": 84,
    "mouth_left": 90,
}

# 关键点索引不保证在所有导出模型上一致，所以给一个"可校验"的顺序：
# 这里取 66/79 是因为它们落在瞳孔中心，实测最稳
KPS5_ORDER = [66, 79, 54, 84, 90]


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """最小二乘相似变换（含缩放+旋转+平移），避免引入 scikit-image 依赖。"""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num
    d = np.ones(dim, dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1.0

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return T
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """返回 2x3 仿射矩阵。

    模板缩放遵循 insightface 约定：
      · image_size 是 112 的整数倍 → 等比放大，无偏移
      · 其他尺寸（128/160/192/224） → 以 128 为基准等比缩放，并右移 8*ratio
        （8 像素来自 96x112 MTCNN 模板与 112x112 模板的差值）
    早期版本这里对 192 也用了 112 等比 + 补边，导致对齐图被压扁 40%，
    106 点反投影后整体缩小、散落在脸外。
    """
    assert landmark.shape == (5, 2), f"需要 5 点，实际 {landmark.shape}"
    dst = ARC_TEMPLATE_112.copy()
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst *= ratio
    dst[:, 0] += diff_x
    M = _umeyama(landmark, dst)
    return M[0:2, :]


def norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """按 5 点对齐裁剪。"""
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(
        img,
        M,
        (image_size, image_size),
        borderValue=0.0,
        flags=cv2.INTER_LINEAR,
    )


def kps5_from_106(pts106: np.ndarray) -> np.ndarray:
    """从 106 点中抽取 ArcFace 需要的 5 点。"""
    return np.asarray([pts106[i] for i in KPS5_ORDER], dtype=np.float32)


def landmark_bbox(pts: np.ndarray, scale: float = 1.20) -> np.ndarray:
    """用关键点外接框（而非检测框）估算人脸区域，用于活体模型裁剪。

    关键点框比检测框更"贴脸"：检测框在侧脸时会带上大片耳朵和背景，
    而活体模型对背景非常敏感。
    """
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    w = max(w, h)  # 取正方形，因为后续 resize 成 1:1
    h = w
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
