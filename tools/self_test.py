# -*- coding: utf-8 -*-
"""无摄像头也能跑的自测：用图片序列冒充摄像头，验证整条流水线。

    python tools/self_test.py                 # 用 dataset/ 里的照片循环播放
    python tools/self_test.py --frames 120 --save out.jpg

检查点：
  · 检测/关键点/识别/活体 是否全部出结果
  · 追踪 ID 是否稳定（不因单帧抖动而跳变）
  · 风控状态是否落在 PASSED
  · 中文标注是否正常渲染（输出图片可直接查看）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from engine import FaceEngine
from fps import FPS


class ImageSequenceCamera:
    """最小可用的摄像头替身：接口与 ThreadedCamera 对齐。"""

    def __init__(self, paths: list[Path], width: int = 1280, height: int = 720, repeat: int = 10):
        self.frames = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # 等比缩放 + 边界复制填边。注意这里**不能**用黑色填充：
            # 黑边会让活体模型把画面当成"屏幕边框"，真人也会被判成攻击
            h, w = img.shape[:2]
            s = min(width / w, height / h)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            canvas = cv2.copyMakeBorder(
                img,
                (height - img.shape[0]) // 2,
                height - img.shape[0] - (height - img.shape[0]) // 2,
                (width - img.shape[1]) // 2,
                width - img.shape[1] - (width - img.shape[1]) // 2,
                cv2.BORDER_REPLICATE,
            )
            self.frames.append(canvas)
        if not self.frames:
            raise SystemExit("dataset/ 里没有可用图片，无法自测")
        self.repeat = repeat
        self.idx = 0
        self.count = 0
        self.dropped = 0
        self.backend = "IMAGE-SEQ"
        self.fps = FPS(window=30)

    def read(self, timeout: float = 0.5, latest: bool = True):
        frame = self.frames[(self.count // self.repeat) % len(self.frames)]
        self.count += 1
        self.fps.tick()
        time.sleep(0.02)  # 模拟 50fps 采集
        return frame

    def stop(self):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=90, help="跑多少帧后停止")
    ap.add_argument("--save", default="outputs/self_test.jpg", help="保存一张带标注的画面")
    ap.add_argument("--repeat", type=int, default=10, help="每张图片重复几帧")
    args = ap.parse_args()

    paths = sorted(p for p in (ROOT / "dataset").glob("*.jpg") if not p.name.startswith("_"))
    if not paths:
        print("dataset/ 里没有 jpg 照片")
        return 1

    cam = ImageSequenceCamera(paths, repeat=args.repeat)
    print(f"图片源: {[p.name for p in paths]}")
    engine = FaceEngine(Config).start(camera=cam)

    seen_ids, states = set(), []
    deadline = time.time() + 180
    while time.time() < deadline:
        st = engine.status()
        if st["faces"]:
            seen_ids.add(st["faces"][0]["track_id"])
            states.append(st["faces"][0]["state"])
        if st["frames"] >= args.frames:
            break
        time.sleep(0.05)
    engine.stop()

    st = engine.status()
    from collections import Counter

    print("\n================ 自测结果 ================")
    print(f"推理帧数    : {st['frames']}   推理帧率: {st['fps']}")
    print(f"设备        : {st['device']}  ({st['device_reason'] or 'GPU 可用'})")
    print(f"分阶段耗时  : {st['infer_ms']}")
    print(f"人脸库      : {st['gallery']['identities']} / {st['gallery']['samples']} 条")
    print(f"风控计数    : {st['counters']}")
    print(f"追踪 ID     : {sorted(seen_ids) or '无'}  （数量越少越稳定）")
    print(f"状态分布    : {dict(Counter(states))}")
    print(f"最近状态    : {st['faces'][0] if st['faces'] else '无人脸'}")

    jpeg = engine.get_jpeg()
    if jpeg:
        out = ROOT / args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(jpeg)
        print(f"标注截图    : {out}")

    ok = st["faces"] and st["faces"][0]["state"] == "passed"
    print("\n判定:", "PASS ✅ 全链路正常" if ok else "FAIL ❌ 状态不是 passed，检查阈值或照片")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
