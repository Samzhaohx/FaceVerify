# -*- coding: utf-8 -*-
"""图片读写封装：解决 Windows 下的非 ASCII 路径、HEIC 格式与 EXIF 方向。

`cv2.imread` / `cv2.imwrite` 在 Windows 上用 narrow-char 的文件接口打开文件，
路径里一旦出现中文（例如 `D:\\4-开发项目\\...`）就直接返回 None / False，
而且**不报错**，只会在 stderr 打一行 warn——很容易被当成"图片损坏"或"目录为空"。

统一改用 `np.fromfile` + `cv2.imdecode`、`cv2.imencode` + `tofile` 后，路径由
Python 的文件对象处理，中文、空格、方括号都能正常读写。
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

HEIC_EXTS = {".heic", ".heif"}


def _read_heic(path: Path) -> np.ndarray | None:
    try:
        import pillow_heif
        from PIL import Image, ImageOps

        pillow_heif.register_heif_opener()
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:  # noqa: BLE001
        log.warning("HEIC 读取失败 %s: %s", path.name, e)
        return None


def imread_any(path: str | Path) -> np.ndarray | None:
    """读图，兼容非 ASCII 路径、HEIC 与 EXIF 方向。失败返回 None。"""
    path = Path(path)

    if path.suffix.lower() in HEIC_EXTS:
        return _read_heic(path)

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as e:
        log.warning("读取失败 %s: %s", path.name, e)
        return None
    if data.size == 0:
        return None

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        # 手机照片常带 EXIF 旋转，imdecode 不处理，交给 PIL 补一次
        try:
            from PIL import Image, ImageOps

            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            return None
    return img


def imwrite_any(path: str | Path, img: np.ndarray, quality: int = 95) -> bool:
    """写图，兼容非 ASCII 路径。按扩展名选择编码格式，默认 jpg。"""
    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except OSError as e:
        log.warning("写入失败 %s: %s", path.name, e)
        return False
