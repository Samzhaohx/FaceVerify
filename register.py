# -*- coding: utf-8 -*-
"""批量录入人脸特征库 —— 对应原项目里"准备 dataset 目录"的那一步。

用法：
    python register.py                        # 扫描 dataset/，按文件名解析人名
    python register.py --name sam             # 全部照片都算同一人（单人多张推荐）
    python register.py --dataset face --clean # 指定目录并重建
    python register.py --visualize            # 额外导出对齐后的人脸，便于肉眼检查

文件名约定：
    sam.jpg / sam_1.jpg / sam-2.jpg / sam(3).jpg  → 都解析为 "sam"
    IMG_0063.jpg                                  → 解析为 "IMG"（所以推荐加 --name）
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algo.align import norm_crop
from algo.imgio import imread_any, imwrite_any
from algo.detector import SCRFDDetector
from algo.recognizer import ArcFaceRecognizer, FaceGallery
from config import Config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def parse_name(stem: str) -> str:
    """sam_1 → sam；张三-02 → 张三；IMG_0063 → IMG"""
    name = re.sub(r"[_\-\s]*\(\d+\)$", "", stem)          # sam(1)
    name = re.sub(r"[_\-\s]+\d+$", "", name)               # sam_1 / sam-2
    name = name.strip()
    return name or stem


def main() -> int:
    ap = argparse.ArgumentParser(description="批量录入人脸特征库")
    ap.add_argument("--dataset", default=None, help="照片目录，默认 config.DATASET_DIR")
    ap.add_argument("--output", default=None, help="pkl 输出路径，默认 config.ENCODING_FILE")
    ap.add_argument("--name", default=None, help="强制所有人共用一个人名（单人照片推荐）")
    ap.add_argument("--clean", action="store_true", help="先清空旧库")
    ap.add_argument("--visualize", action="store_true", help="导出对齐后的 112x112 人脸")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 推理")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config
    dataset = Path(args.dataset) if args.dataset else cfg.DATASET_DIR
    output = Path(args.output) if args.output else cfg.ENCODING_FILE
    prefer_gpu = cfg.PREFER_GPU and not args.cpu

    if not dataset.exists():
        logging.error("照片目录不存在: %s", dataset)
        return 1

    paths = sorted(p for p in dataset.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        logging.error("%s 里没有找到图片（支持 %s）", dataset, ", ".join(sorted(IMG_EXTS)))
        return 1

    logging.info("加载模型 ...")
    det = SCRFDDetector(str(cfg.MODEL_DIR / cfg.DET_MODEL), det_thresh=cfg.DET_THRESHOLD, prefer_gpu=prefer_gpu)
    rec = ArcFaceRecognizer(str(cfg.MODEL_DIR / cfg.RECOGNITION_MODEL), prefer_gpu=prefer_gpu)

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        if p.name.startswith("_"):     # 跳过调试可视化产物
            continue
        groups[args.name or parse_name(p.stem)].append(p)

    saved_dir = output.parent / "aligned"
    if args.visualize:
        saved_dir.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    names: list[str] = []
    sources: list[str] = []
    failed: list[str] = []

    for name, files in sorted(groups.items()):
        logging.info("\n→ %s (%d 张)", name, len(files))
        person_embs = []
        for f in files:
            img = imread_any(f)
            if img is None:
                logging.warning("   ✗ %s 读取失败", f.name)
                failed.append(f.name)
                continue
            faces = det.detect(img)
            if not faces:
                logging.warning("   ✗ %s 未检出人脸（建议换清晰正脸照）", f.name)
                failed.append(f.name)
                continue
            faces.sort(key=lambda x: x.area, reverse=True)
            if len(faces) > 1:
                logging.warning("   ! %s 检出 %d 张脸，取最大的一张", f.name, len(faces))
            face = faces[0]
            if face.width < cfg.MIN_FACE_SIZE:
                logging.warning("   ✗ %s 人脸过小(%dpx)，跳过", f.name, face.width)
                failed.append(f.name)
                continue

            emb = rec.embed_from_face(img, face.kps)
            person_embs.append(emb)
            embeddings.append(emb)
            names.append(name)
            sources.append(f.name)

            if args.visualize:
                crop = norm_crop(img, face.kps, 112)
                out = saved_dir / f"{name}_{f.stem}_aligned.jpg"
                imwrite_any(out, crop)
            logging.info("   ✓ %s  det=%.2f  框=%dpx", f.name, face.score, face.width)

        if len(person_embs) >= 2:
            m = np.array(person_embs) @ np.array(person_embs).T
            off = m[~np.eye(len(person_embs), dtype=bool)]
            flag = "⚠️ 偏低，检查是否混入不同的人/侧脸" if off.min() < 0.35 else "正常"
            logging.info("   同人相似度 min=%.3f mean=%.3f → %s", off.min(), off.mean(), flag)

    if not embeddings:
        logging.error("没有成功录入任何照片")
        return 1

    if args.clean and output.exists():
        output.unlink()
    gallery = FaceGallery(output, np.vstack(embeddings), names, sources)
    gallery.save()

    # 不同人之间的相似度：过高说明特征区分度不足或录入了同一个人
    ids = sorted(set(names))
    if len(ids) >= 2:
        cent = {i: np.mean([e for e, n in zip(embeddings, names) if n == i], axis=0) for i in ids}
        cent = {k: v / np.linalg.norm(v) for k, v in cent.items()}
        logging.info("\n不同人之间的最大相似度：")
        worst = (0.0, "", "")
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                s = float(cent[a] @ cent[b])
                logging.info("  %s ↔ %s : %.3f", a, b, s)
                if s > worst[0]:
                    worst = (s, a, b)
        if worst[0] > cfg.SIM_THRESHOLD:
            logging.warning("  ⚠️ %s 与 %s 相似度 %.3f 已超过阈值 %.2f，可能撞脸",
                            worst[1], worst[2], worst[0], cfg.SIM_THRESHOLD)

    logging.info("\n完成：%d 人 / %d 条特征 → %s", len(ids), len(embeddings), output)
    if failed:
        logging.info("跳过 %d 张：%s", len(failed), ", ".join(failed))
    logging.info("现在启动服务：python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
