# -*- coding: utf-8 -*-
"""实时人脸核验引擎：检测 → 关键点 → 追踪 → 识别 → 活体 → 风控分级。

线程模型：
    [采集线程] --帧队列(有界,丢旧帧)--> [推理线程] --最新JPEG--> [Flask 推流/状态接口]
推理线程内部再按 DETECT_INTERVAL 做跳帧检测，非检测帧只跑卡尔曼预测，
从而把"检测+识别"的重计算摊到多帧上。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from algo.detector import SCRFDDetector
from algo.landmark import Landmark106
from algo.liveness import LivenessDetector, diagnose
from algo.ort_utils import device_info
from algo.recognizer import ArcFaceRecognizer, FaceGallery
from algo.tracker import ByteTracker, Track
from camera import ThreadedCamera
from config import Config
from draw import Annotator
from fps import FPS, StageTimer

log = logging.getLogger(__name__)


# ---------- 风控状态 ----------

STATE_PASSED = "passed"
STATE_SPOOF = "spoof"
STATE_MISMATCH = "mismatch"
STATE_UNKNOWN = "unknown"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

STATE_TEXT = {
    STATE_PASSED: "验证通过",
    STATE_SPOOF: "活体异常",
    STATE_MISMATCH: "身份不匹配",
    STATE_UNKNOWN: "未知人员",
}


@dataclass
class FaceResult:
    track_id: int
    bbox: np.ndarray
    state: str = STATE_UNKNOWN
    name: str | None = None
    similarity: float = 0.0
    liveness: float = 0.0
    det_score: float = 0.0
    kps106: np.ndarray | None = None
    reason: str = ""                      # 一句话结论（画面上显示）
    detail: list[str] = field(default_factory=list)  # 可读原因列表（面板显示）
    color: tuple = (235, 235, 235)

    @property
    def state_text(self) -> str:
        return STATE_TEXT.get(self.state, "未知")

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state,
            "state_text": self.state_text,
            "name": self.name,
            "similarity": round(float(self.similarity), 3),
            "liveness": round(float(self.liveness), 3),
            "det_score": round(float(self.det_score), 3),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "reason": self.reason,
            "detail": list(self.detail),
        }


@dataclass
class EngineStats:
    frames: int = 0
    infer_ms: dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: {STATE_PASSED: 0, STATE_SPOOF: 0, STATE_MISMATCH: 0, STATE_UNKNOWN: 0})
    started_at: float = field(default_factory=time.time)


class FaceEngine:
    def __init__(self, cfg=Config):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.camera: ThreadedCamera | None = None

        log.info("加载模型 ...")
        self.detector = SCRFDDetector(
            str(cfg.MODEL_DIR / cfg.DET_MODEL),
            det_size=tuple(cfg.DET_SIZE) if isinstance(cfg.DET_SIZE, (tuple, list)) else (cfg.DET_SIZE, cfg.DET_SIZE),
            det_thresh=cfg.DET_THRESHOLD,
            nms_thresh=cfg.DET_NMS_THRESHOLD,
            prefer_gpu=cfg.PREFER_GPU,
        )
        self.landmark = Landmark106(str(cfg.MODEL_DIR / cfg.LANDMARK_MODEL), prefer_gpu=cfg.PREFER_GPU)
        self.recognizer = ArcFaceRecognizer(str(cfg.MODEL_DIR / cfg.RECOGNITION_MODEL), prefer_gpu=cfg.PREFER_GPU)
        self.liveness = LivenessDetector(str(cfg.MODEL_DIR / cfg.LIVENESS_MODEL), prefer_gpu=cfg.PREFER_GPU)
        self.tracker = ByteTracker(
            iou_thresh=cfg.TRACK_IOU_THRESHOLD,
            high_thresh=cfg.TRACK_HIGH_THRESHOLD,
            low_thresh=cfg.TRACK_LOW_THRESHOLD,
            max_age=cfg.TRACK_MAX_AGE,
            min_hits=cfg.TRACK_MIN_HITS,
        )
        self.annotator = Annotator(cfg)

        self.gallery = FaceGallery.load(cfg.ENCODING_FILE)
        if len(self.gallery) == 0:
            log.warning("人脸库为空，所有人都会被判为「未知人员」。先运行: python register.py")

        self.fps = FPS(window=30)                 # 推理/推流帧率
        self.timer = StageTimer(("detect", "landmark", "recognize", "liveness", "draw"))
        self._encode_ms = 0.0
        self.stats = EngineStats()

        self._jpeg: bytes | None = None
        self._results: list[FaceResult] = []
        self._frame_size = (0, 0)
        self._frame_idx = 0
        self._last_error = ""
        self._pending_cache: tuple[float, int] | None = None

    # ---------------- 生命周期 ----------------

    def start(self, camera=None) -> "FaceEngine":
        """camera 参数用于测试：传入任何带 read()/stop() 的对象即可替换真实摄像头。"""
        if camera is not None:
            self.camera = camera
        else:
            self.camera = ThreadedCamera(
                index=self.cfg.CAMERA_INDEX,
                width=self.cfg.FRAME_WIDTH,
                height=self.cfg.FRAME_HEIGHT,
                fps=self.cfg.CAMERA_FPS,
                backend=self.cfg.CAMERA_BACKEND,
                queue_size=self.cfg.FRAME_QUEUE_SIZE,
            ).start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()
        log.info("引擎已启动 | 推理设备=%s | 人脸库=%d 条/%d 人",
                 device_info()["device"], len(self.gallery), len(self.gallery.identities))
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.camera:
            self.camera.stop()

    # ---------------- 推理主循环 ----------------

    def _loop(self) -> None:
        while self._running:
            assert self.camera is not None
            frame = self.camera.read(timeout=0.5, latest=True)
            if frame is None:
                continue
            try:
                annotated, results = self._process(frame)
            except Exception as e:  # noqa: BLE001 - 单帧异常不能打死整条流水线
                log.exception("推理异常: %s", e)
                self._last_error = f"{type(e).__name__}: {e}"
                continue

            q = self.cfg.JPEG_QUALITY
            t_enc = time.perf_counter()
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
            enc_ms = (time.perf_counter() - t_enc) * 1000.0
            # 编码发生在画完 HUD 之后，所以单独统计，回填到下一帧的状态里
            self._encode_ms = enc_ms if self._encode_ms == 0 else 0.85 * self._encode_ms + 0.15 * enc_ms
            self.fps.tick()
            with self._lock:
                if ok:
                    self._jpeg = buf.tobytes()
                self._results = results
                self._frame_size = (annotated.shape[1], annotated.shape[0])
                self.stats.frames += 1
                self.stats.infer_ms = {**self.timer.values, "encode": round(self._encode_ms, 2)}

    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, list[FaceResult]]:
        self._frame_idx += 1
        idx = self._frame_idx
        do_detect = (idx % max(1, self.cfg.DETECT_INTERVAL) == 0) or self._frame_idx == 1

        if do_detect:
            with self.timer.start("detect"):
                faces = self.detector.detect(frame)
            # 太小的脸直接丢：活体和识别都不可靠，留着只会产生误判
            faces = [f for f in faces if f.width >= self.cfg.MIN_FACE_SIZE]
            # 单人核验模式：只处理面积最大（离镜头最近）的那张脸
            if self.cfg.SINGLE_FACE and faces:
                faces = faces[:1]

            with self.timer.start("landmark"):
                for f in faces:
                    self.landmark.refine(frame, f)

            tracks = self.tracker.update(faces)
        else:
            tracks = self.tracker.predict_only()

        if self.cfg.SINGLE_FACE and tracks:
            tracks = [max(tracks, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))]

        results: list[FaceResult] = []
        for tr in tracks:
            res = self._evaluate(frame, tr, idx)
            if res is not None:
                results.append(res)

        hud = self._hud(len(results))

        # 镜像预览：只翻"给人看的那一帧"，检测/识别/活体始终跑在原图上。
        # 为什么不在采集端就翻：
        #   1) ArcFace 不是严格翻转不变 —— 实测同一张脸镜像后与库内相似度
        #      从 1.000 掉到 0.909~0.949，白丢精度；
        #   2) 翻转后再画字，文字会变成反的。
        # 所以这里翻转像素 + 把框的 x 坐标做镜像映射，文字保持正向可读。
        if self.cfg.MIRROR:
            h, w = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            for r in results:
                x1, y1, x2, y2 = [float(v) for v in r.bbox]
                r.bbox = np.array([w - 1 - x2, y1, w - 1 - x1, y2], dtype=np.float32)
                if r.kps106 is not None:
                    k = r.kps106.copy()
                    k[:, 0] = w - 1 - k[:, 0]
                    r.kps106 = k

        with self.timer.start("draw"):
            annotated = self.annotator.draw(
                frame, results, hud=hud if self.cfg.SHOW_HUD else None,
                show_keypoints=self.cfg.SHOW_KEYPOINTS,
            )
        return annotated, results

    # ---------------- 单张脸的判定 ----------------

    def _evaluate(self, frame: np.ndarray, tr: Track, idx: int) -> FaceResult | None:
        c = self.cfg
        cache = tr.cache
        res = FaceResult(track_id=tr.track_id, bbox=tr.bbox.astype(np.float32), det_score=tr.score, kps106=tr.kps106)

        # ---- 1. 识别（带缓存：同一 track 不必每帧重算 ArcFace）----
        need_emb = (
            "emb" not in cache
            or tr.kps is None
            or (idx - cache.get("emb_frame", -10**9)) >= c.RECOGNIZE_INTERVAL
            or cache.get("miss", 0) > 0
        )
        if need_emb and tr.kps is not None:
            with self.timer.start("recognize"):
                emb = self.recognizer.embed_from_face(frame, tr.kps)
                # 同一轨迹内做特征 EMA：摄像头噪声、轻微姿态变化会让单帧相似度抖动
                # 3~5 个百分点，融合后阈值附近的误判明显减少（见 docs/DEVNOTES.md 第 12 条）
                prev_emb = cache.get("emb_ema")
                if prev_emb is None or c.EMBEDDING_SMOOTH >= 1.0:
                    emb_use = emb
                else:
                    w = float(c.EMBEDDING_SMOOTH)
                    fused = (1.0 - w) * prev_emb + w * emb
                    n = np.linalg.norm(fused)
                    emb_use = fused / n if n > 0 else emb
                match = self.gallery.match(emb_use)
            cache["emb"] = emb
            cache["emb_ema"] = emb_use
            cache["match"] = (match.name, match.score, match.second_name, match.second_score)
            cache["emb_frame"] = idx

        name, sim = (cache["match"][0], cache["match"][1]) if "match" in cache else (None, -1.0)

        # ---- 2. 静默活体（EMA 平滑，抑制单帧抖动）----
        need_live = (
            "live" not in cache
            or (idx - cache.get("live_frame", -10**9)) >= c.LIVENESS_INTERVAL
            or cache.get("miss", 0) > 0
        )
        if need_live:
            with self.timer.start("liveness"):
                live_now, live_dbg = self.liveness.predict(frame, self._liveness_box(tr))
            prev = cache.get("live")
            cache["live"] = live_now if prev is None else (c.LIVENESS_EMA * prev + (1 - c.LIVENESS_EMA) * live_now)
            cache["live_frame"] = idx
            cache["live_raw"] = live_now
            cache["live_dbg"] = live_dbg

        live = float(cache.get("live", 0.0))
        live_dbg = cache.get("live_dbg", {})

        # ---- 3. 风控分级 ----
        res.name = name if (sim >= c.SIM_THRESHOLD) else None
        res.similarity = 0.0 if sim < 0 else float(sim)
        res.liveness = live

        if len(self.gallery) == 0:
            res.state = STATE_UNKNOWN
            res.reason = "人脸库为空"
            res.detail = ["还没有录入任何人脸，先运行 python register.py"]
        elif tr.hits < c.IDENTITY_CONFIRM_HITS:
            res.state = STATE_UNKNOWN
            res.reason = "身份确认中"
            res.detail = [f"该目标只连续出现 {tr.hits} 帧，凑够 {c.IDENTITY_CONFIRM_HITS} 帧才出结论"]
        elif live < c.LIVENESS_THRESHOLD:
            res.state = STATE_SPOOF
            res.detail = diagnose(live, live_dbg, c.LIVENESS_THRESHOLD, face_px=int(tr.bbox[2] - tr.bbox[0]))
            # 画面上只写"原因"，分数已经在上方那行显示过了，不重复占地方
            res.reason = res.detail[0] if res.detail else "活体检测未通过"
        elif sim >= c.SIM_THRESHOLD:
            res.state, res.reason = STATE_PASSED, ""
        elif sim >= c.MISMATCH_THRESHOLD:
            res.state = STATE_MISMATCH
            res.reason = f"最相似 {sim:.2f} < 阈值 {c.SIM_THRESHOLD}"
            res.detail = [
                f"最相近的是「{name}」，相似度 {sim:.2f}，低于 {c.SIM_THRESHOLD}",
                f"差距 {c.SIM_THRESHOLD - sim:.2f}，若确为本人可下调 SIM_THRESHOLD 或补录该角度照片",
            ]
        else:
            res.state, res.reason = STATE_UNKNOWN, "库内无此人"
            res.detail = [
                f"与库内最相似的人也只有 {max(sim, 0):.2f}（阈值 {c.SIM_THRESHOLD}）",
                "若此人应被识别，请把照片加入 dataset/ 后重新运行 register.py",
            ]

        res.color = {
            STATE_PASSED: c.COLOR_PASSED,
            STATE_SPOOF: c.COLOR_SPOOF,
            STATE_MISMATCH: c.COLOR_MISMATCH,
            STATE_UNKNOWN: c.COLOR_UNKNOWN,
        }[res.state]

        with self._lock:
            self.stats.counters[res.state] = self.stats.counters.get(res.state, 0) + 1
        return res

    @staticmethod
    def _liveness_box(tr: Track) -> np.ndarray:
        """优先用 106 点外接框；退化时用追踪框。"""
        if tr.kps106 is not None:
            from algo.align import landmark_bbox

            return landmark_bbox(tr.kps106, 1.0)
        return tr.bbox

    # ---------------- HUD ----------------

    def _hud(self, n_faces: int) -> dict:
        dev = device_info()
        lines = [
            f"设备 {dev['device']}   ONNXRuntime {dev['onnxruntime']}",
            f"全链路 {self.timer.total + self._encode_ms:.0f} ms   {self._frame_size[0]}x{self._frame_size[1]}",
            f"人脸库 {len(self.gallery.identities)} 人 / {len(self.gallery)} 条",
            f"丢帧 {self.camera.dropped if self.camera else 0}   目标 {n_faces}",
        ]
        return {"fps_text": f"{self.fps.value:.1f} FPS", "lines": lines}

    # ---------------- 对外接口 ----------------

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        with self._lock:
            results = [r.to_dict() for r in self._results]
            counters = dict(self.stats.counters)
            frames = self.stats.frames
            infer = dict(self.stats.infer_ms)
            size = self._frame_size
        dev = device_info()
        return {
            "ok": True,
            "fps": round(self.fps.value, 1),
            "infer_ms": infer,
            "frames": frames,
            "uptime": round(time.time() - self.stats.started_at, 1),
            "device": dev["device"],
            "device_reason": dev["reason"],
            "providers": dev["providers"],
            "frame_size": list(size),
            "mode": "single" if self.cfg.SINGLE_FACE else "multi",
            "gallery": {
                "identities": self.gallery.identities,
                "samples": len(self.gallery),
                "pending": self.pending_photos(),
            },
            "faces": results,
            "counters": counters,
            "thresholds": {
                "similarity": self.cfg.SIM_THRESHOLD,
                "mismatch": self.cfg.MISMATCH_THRESHOLD,
                "liveness": self.cfg.LIVENESS_THRESHOLD,
            },
            "camera": {
                "index": str(self.cfg.CAMERA_INDEX),
                "backend": self.camera.backend if self.camera else "",
                "dropped": self.camera.dropped if self.camera else 0,
                "capture_fps": round(self.camera.fps.value, 1) if self.camera else 0,
                "negotiated_fps": getattr(self.camera, "negotiated_fps", 0) if self.camera else 0,
                "mirror": bool(self.cfg.MIRROR),
            },
            "error": self._last_error,
        }

    def reload_gallery(self) -> int:
        self.gallery = FaceGallery.load(self.cfg.ENCODING_FILE)
        self.tracker.reset()
        self._pending_cache = None
        log.info("人脸库已重载: %d 人", len(self.gallery.identities))
        return len(self.gallery)

    def pending_photos(self) -> int:
        """dataset/ 里比 faces.pkl 更新的照片数。

        >0 说明有新照片没跑 register.py —— "重载"只重新读 pkl，不会提特征，
        所以这种情况重载多少次都没用。页面会据此给出提示。
        """
        now = time.time()
        if self._pending_cache and now - self._pending_cache[0] < 2.0:
            return self._pending_cache[1]

        ds, pkl = Path(self.cfg.DATASET_DIR), Path(self.cfg.ENCODING_FILE)
        count = 0
        if ds.exists():
            base = pkl.stat().st_mtime if pkl.exists() else 0.0
            for p in ds.iterdir():
                if p.name.startswith("_") or p.suffix.lower() not in _IMG_EXTS:
                    continue
                try:
                    if p.stat().st_mtime > base:
                        count += 1
                except OSError:
                    continue
        self._pending_cache = (now, count)
        return count
