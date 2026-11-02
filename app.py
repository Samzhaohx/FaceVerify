# -*- coding: utf-8 -*-
"""Flask Web 服务：MJPEG 推流 + 状态接口（沿用原项目的推流与前端交互方式）。

    python app.py            启动服务（默认 0.0.0.0:5000）
    python app.py --port 8000 --index 1
"""
from __future__ import annotations

import argparse
import logging
import socket
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from camera import CameraError
from config import Config
from engine import FaceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("app")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

_engine: FaceEngine | None = None
_engine_error: str = ""
_lock = threading.Lock()


def get_engine() -> FaceEngine | None:
    """懒加载单例。用锁避免 Flask 多线程下重复初始化模型。"""
    global _engine, _engine_error
    with _lock:
        if _engine is not None:
            return _engine
        if _engine_error:
            return None
        try:
            _engine = FaceEngine(Config).start()
        except CameraError as e:
            _engine_error = str(e)
            log.error("摄像头初始化失败：%s", e)
        except Exception as e:  # noqa: BLE001
            _engine_error = f"{type(e).__name__}: {e}"
            log.exception("引擎初始化失败")
        return _engine


def _placeholder_jpeg(text: str, w: int = 960, h: int = 540) -> bytes:
    """没有画面时返回一张提示图，避免前端出现破图。"""
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    from draw import Annotator

    ann = Annotator(Config)
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f1 = ann.font(26)
        f2 = ann.font(16)
        d.text((w // 2 - 130, h // 2 - 40), "等待摄像头画面", font=f1, fill=(230, 230, 230))
        for i, line in enumerate(text.split("\n")[:6]):
            d.text((40, h // 2 + 10 + i * 24), line, font=f2, fill=(160, 160, 165))
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        cv2.putText(img, "NO SIGNAL", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


def _mjpeg_generator():
    engine = get_engine()
    interval = 1.0 / max(1, Config.STREAM_FPS_LIMIT)
    while True:
        t0 = time.perf_counter()
        if engine is None:
            jpeg = _placeholder_jpeg(_engine_error or "引擎未启动")
            yield _part(jpeg)
            time.sleep(1.0)
            continue
        jpeg = engine.get_jpeg()
        if jpeg is None:
            jpeg = _placeholder_jpeg("模型与摄像头正在初始化 ...")
        yield _part(jpeg)
        dt = time.perf_counter() - t0
        if dt < interval:
            time.sleep(interval - dt)


def _part(jpeg: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"


# ---------------- 路由 ----------------


@app.route("/")
def index():
    engine = get_engine()
    return render_template(
        "index.html",
        error=_engine_error,
        mode="单人核验" if Config.SINGLE_FACE else "多人识别",
        status_interval=Config.STATUS_INTERVAL_MS,
        gallery=engine.gallery.identities if engine else [],
        thresholds={
            "sim": Config.SIM_THRESHOLD,
            "live": Config.LIVENESS_THRESHOLD,
            "mismatch": Config.MISMATCH_THRESHOLD,
        },
    )


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    engine = get_engine()
    if engine is None:
        return jsonify({"ok": False, "error": _engine_error or "引擎未启动", "faces": []})
    return jsonify(engine.status())


@app.route("/api/reload", methods=["POST"])
def api_reload():
    engine = get_engine()
    if engine is None:
        return jsonify({"ok": False, "error": _engine_error}), 503
    n = engine.reload_gallery()
    return jsonify({"ok": True, "identities": len(engine.gallery.identities), "samples": n})


@app.route("/api/health")
def api_health():
    return jsonify({"ok": _engine is not None, "error": _engine_error})


def _lan_ip() -> str:
    """拿本机在局域网里的 IP，方便直接告诉用户用手机访问哪个地址。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser(description="实时人脸核验 + 静默活体 Web 服务")
    ap.add_argument("--host", default=Config.HOST)
    ap.add_argument("--port", type=int, default=Config.PORT)
    ap.add_argument("--index", default=None, help="摄像头序号，覆盖 config.CAMERA_INDEX")
    ap.add_argument("--no-gpu", action="store_true", help="强制 CPU 推理")
    ap.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="水平镜像预览（默认跟随 config.MIRROR）；--no-mirror 关闭",
    )
    args = ap.parse_args()

    if args.index is not None:
        Config.CAMERA_INDEX = int(args.index) if str(args.index).isdigit() else args.index
    if args.no_gpu:
        Config.PREFER_GPU = False
    if args.mirror is not None:
        Config.MIRROR = args.mirror

    # 提前初始化，让摄像头/模型问题在启动日志里就暴露，而不是等第一个请求
    get_engine()

    ip = _lan_ip()
    log.info("=" * 62)
    log.info("本机访问:   http://127.0.0.1:%d", args.port)
    log.info("局域网访问: http://%s:%d   （手机/平板同一 WiFi 直接打开）", ip, args.port)
    log.info("=" * 62)

    # use_reloader=False：避免调试重载把摄像头打开两次
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
