# -*- coding: utf-8 -*-
"""多线程摄像头采集（沿用原项目方案，补齐了 Windows 后端选择与断线重连）。

核心问题：cv2.VideoCapture.read() 是阻塞调用，按 30fps 算一次要 33ms。
如果在 Flask 的推流生成器里直接 read()，那么"采集 → 推理 → 编码"会串成一条链，
任何一环慢都会拖垮帧率，而且多设备访问时每个客户端各 read 一次会互相抢摄像头。

解决：采集单独一条线程，只管把最新帧塞进有界队列；推理线程从队列取帧；
队列满时丢最旧帧（宁可丢帧也不要积压延迟）。
"""
from __future__ import annotations

import logging
import platform
import queue
import threading
import time

import cv2
import numpy as np

from fps import FPS

log = logging.getLogger(__name__)


def _candidate_backends() -> list[int | None]:
    """Windows 下 MSMF 打开慢且对 MJPG 支持差，优先 DSHOW。"""
    if platform.system() == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    if platform.system() == "Darwin":
        return [cv2.CAP_AVFOUNDATION, None]
    return [cv2.CAP_V4L2, None]


class CameraError(RuntimeError):
    pass


class ThreadedCamera:
    def __init__(
        self,
        index: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: int | None = None,
        queue_size: int = 2,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self.fps = FPS(window=30)
        self.dropped = 0
        self.read_failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._backend_used = ""
        self.negotiated_fps = 0.0

        self._open(backend)

    # ---------------- 打开设备 ----------------

    def _open(self, backend: int | None) -> None:
        backends = [backend] if backend is not None else _candidate_backends()
        errors = []
        for be in backends:
            name = self._backend_name(be)
            cap = cv2.VideoCapture(self.index, be) if be is not None else cv2.VideoCapture(self.index)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{name}: 无法打开")
                continue
            # MJPG：不设置的话很多 UVC 摄像头在 1280x720 下只能跑 5~10fps（YUY2 带宽限制）
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            # 内部缓冲压到 1，否则 read() 会拿到几帧之前的老画面
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                errors.append(f"{name}: 打开成功但读不到帧")
                continue

            self._cap = cap
            self._backend_used = name
            real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
            real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
            real_fps = cap.get(cv2.CAP_PROP_FPS) or 0
            self.negotiated_fps = float(real_fps)
            log.info("摄像头已打开: index=%s backend=%s %dx%d @%.0ffps", self.index, name, real_w, real_h, real_fps)
            if real_fps and real_fps < self.target_fps * 0.8:
                log.warning(
                    "摄像头协商帧率 %.0f 低于请求的 %d：该分辨率下设备可能只支持此帧率",
                    real_fps, self.target_fps,
                )
            self.width, self.height = real_w, real_h
            return

        raise CameraError(
            "无法打开摄像头 index=%s。已尝试: %s。\n"
            "排查：1) 是否被其他程序（微信/钉钉/浏览器）占用  2) Windows 隐私设置是否允许桌面应用访问相机  "
            "3) 换个 CAMERA_INDEX 试试（0/1/2）" % (self.index, "; ".join(errors) or "无可用后端")
        )

    @staticmethod
    def _backend_name(be: int | None) -> str:
        mapping = {
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_AVFOUNDATION: "AVFOUNDATION",
            cv2.CAP_V4L2: "V4L2",
            None: "DEFAULT",
        }
        return mapping.get(be, str(be))

    @property
    def backend(self) -> str:
        return self._backend_used

    # ---------------- 采集线程 ----------------

    def start(self) -> "ThreadedCamera":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        fail_streak = 0
        diagnosed = False
        reopens_left = 1
        t_start = time.time()
        got = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail_streak += 1
                self.read_failures += 1
                if fail_streak == 30:
                    log.warning("连续读帧失败，尝试重新打开摄像头")
                    with self._lock:
                        self._cap.release()
                        try:
                            self._open(None)
                        except CameraError as e:
                            log.error("重开摄像头失败: %s", e)
                            break
                    fail_streak = 0
                time.sleep(0.01)
                continue
            fail_streak = 0
            got += 1
            self._put(frame)

            # 启动后 3 秒做一次帧率体检：
            #   · <2fps  → 设备大概率被别的程序抢占/驱动回退，重开一次
            #   · 明显低于请求 → 该分辨率下设备只支持这个帧率，给出人话提示
            if not diagnosed and time.time() - t_start >= 3.0:
                rate = got / (time.time() - t_start)
                if rate < 2.0 and reopens_left > 0:
                    reopens_left -= 1
                    log.warning("采集帧率仅 %.1f fps，设备可能被其他程序占用，尝试重开一次", rate)
                    with self._lock:
                        self._cap.release()
                        try:
                            self._open(None)
                        except CameraError as e:
                            log.error("重开摄像头失败: %s", e)
                            diagnosed = True
                            continue
                    t_start, got = time.time(), 0
                else:
                    if rate < self.target_fps * 0.6:
                        log.warning(
                            "实际采集 %.1f fps，低于请求的 %d fps。该分辨率下摄像头可能只支持此帧率"
                            "（MJPG/USB 带宽限制），或自动曝光在暗光下拉长了曝光时间。"
                            "可尝试：降低 FRAME_WIDTH/HEIGHT、换 USB3.0 口、补光。",
                            rate, self.target_fps,
                        )
                    diagnosed = True

        with self._lock:
            if self._cap is not None:
                self._cap.release()

    def _put(self, frame: np.ndarray) -> None:
        """有界队列：满了就丢最旧的一帧，保证消费者永远拿到最新画面。"""
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    # ---------------- 消费者接口 ----------------

    def read(self, timeout: float = 0.5, latest: bool = True) -> np.ndarray | None:
        """取一帧。latest=True 时把队列里积压的旧帧全部丢弃，只保留最后一帧。"""
        try:
            frame = self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if latest:
            while True:
                try:
                    frame = self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    break
        self.fps.tick()
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def release(self) -> None:
        self.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
