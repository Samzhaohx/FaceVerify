# -*- coding: utf-8 -*-
"""FPS / 耗时统计（沿用原项目的实时 FPS 展示思路，扩展了分阶段耗时）。"""
from __future__ import annotations

import time
from collections import deque


class FPS:
    """滑动窗口 FPS。用时间戳队列而不是 1/dt，读数更稳。"""

    def __init__(self, window: int = 30):
        self.window = window
        self._ts: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._ts.append(time.perf_counter())

    @property
    def value(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        if span <= 0:
            return 0.0
        return (len(self._ts) - 1) / span

    def reset(self) -> None:
        self._ts.clear()


class StageTimer:
    """分阶段耗时统计（指数滑动平均），用于定位瓶颈到底在检测还是识别。"""

    def __init__(self, names: tuple[str, ...], alpha: float = 0.15):
        self.alpha = alpha
        self._avg = {n: 0.0 for n in names}
        self._t0 = 0.0
        self._cur: str | None = None

    def start(self, name: str) -> "StageTimer":
        self._cur = name
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> None:
        if self._cur is None:
            return
        dt = (time.perf_counter() - self._t0) * 1000.0
        if self._avg[self._cur] == 0:
            self._avg[self._cur] = dt
        else:
            self._avg[self._cur] = (1 - self.alpha) * self._avg[self._cur] + self.alpha * dt
        self._cur = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    @property
    def values(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self._avg.items()}

    @property
    def total(self) -> float:
        return round(sum(self._avg.values()), 2)
