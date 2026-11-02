# -*- coding: utf-8 -*-
"""OnnxRuntime 会话工厂：优先 GPU，失败自动降级 CPU。

两个坑：
1. `onnxruntime-gpu` 装上不代表能用 —— 缺 CUDA/cuDNN 运行时 DLL 时，
   provider 依然出现在 get_available_providers() 里，直到建会话才炸。
2. 让它炸一次要付 ~0.3s/模型 的代价，还会往 stderr 打一坨红色报错。
所以这里先用 ctypes 做一次廉价的运行时预检，不可用就直接走 CPU。
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform

import onnxruntime as ort

log = logging.getLogger(__name__)

_state = {"device": None, "reason": "", "cuda_ok": None, "checked": False}

# ORT 1.31 起需要 CUDA 13 + cuDNN 9；旧版本对应 12/11
_CUDA_RUNTIME_NAMES = {
    "Windows": ["cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt64_11.dll"],
    "Linux": ["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11"],
}


def _cuda_runtime_present() -> bool:
    """检查 CUDA 运行时是否可被动态加载（不依赖 nvidia-smi，也不需要 torch）。"""
    names = _CUDA_RUNTIME_NAMES.get(platform.system(), [])
    for name in names:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def cuda_available() -> tuple[bool, str]:
    """返回 (是否可用, 原因)。结果缓存，避免每个模型都探测一次。"""
    if _state["cuda_ok"] is not None:
        return _state["cuda_ok"], _state["reason"]

    env = os.environ.get("ORT_DEVICE", "").strip().lower()
    if env == "cpu":
        _state["cuda_ok"], _state["reason"] = False, "环境变量 ORT_DEVICE=cpu 强制 CPU"
        return False, _state["reason"]

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        _state["cuda_ok"], _state["reason"] = False, "onnxruntime 未包含 CUDA provider（装的是 CPU 版）"
        return False, _state["reason"]

    if not _cuda_runtime_present():
        _state["cuda_ok"], _state["reason"] = (
            False,
            "未检测到 CUDA 运行时（cublasLt64_*.dll）。装 GPU 版需要 CUDA 13 + cuDNN 9，"
            "或改用 requirements-cpu.txt",
        )
        return False, _state["reason"]

    _state["cuda_ok"], _state["reason"] = True, ""
    return True, ""


def _opts() -> ort.SessionOptions:
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 限制线程数：多个模型串行推理时，线程抢满反而拖低整条流水线吞吐
    o.intra_op_num_threads = 4
    o.inter_op_num_threads = 2
    o.log_severity_level = 3
    return o


def create_session(path: str, prefer_gpu: bool = True, name: str = "") -> ort.InferenceSession:
    """创建推理会话。prefer_gpu 时优先 CUDA，不可用静默降级 CPU。"""
    if prefer_gpu:
        ok, _reason = cuda_available()
    else:
        ok, _reason = False, "配置要求 CPU"

    if ok:
        try:
            sess = ort.InferenceSession(
                path, sess_options=_opts(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            if "CUDAExecutionProvider" in sess.get_providers():
                _state.update(device="CUDA", cuda_ok=True)
                log.info("[%-11s] 后端 = CUDA", name)
                return sess
        except Exception as e:  # noqa: BLE001 - 显存不足/驱动不匹配等，一律降级
            _state.update(cuda_ok=False, reason=f"CUDA 建会话失败: {type(e).__name__}: {e}")
            log.warning("[%-11s] CUDA 建会话失败，降级 CPU: %s", name, e)

    sess = ort.InferenceSession(path, sess_options=_opts(), providers=["CPUExecutionProvider"])
    if _state["device"] is None:
        _state["device"] = "CPU"
    if not _state["reason"]:
        _state["reason"] = _reason
    log.info("[%-11s] 后端 = CPU", name)
    return sess


def device_info() -> dict:
    """给启动日志与 Web 状态面板用的设备信息。"""
    return {
        "device": _state["device"] or "未初始化",
        "reason": _state["reason"],
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
    }
