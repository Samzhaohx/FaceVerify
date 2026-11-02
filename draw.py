# -*- coding: utf-8 -*-
"""画面标注：人脸框、追踪 ID、姓名、活体分数、相似度、FPS，支持中文。

cv2.putText 用 Hershey 矢量字体，画不了中文（会变成一堆"????"）。
这里用 FreeType 字体 + PIL 绘制：一次 BGR→RGB 转换，所有图元在同一张
PIL 图上画完再转回来，比"每个文字都转换一次"快一个数量级。
"""
from __future__ import annotations

import logging
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\Deng.ttf",      # 等线
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    log.warning("未找到中文字体，中文将降级为拼音/英文标注")
    return None


class Annotator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.font_path = find_font(getattr(cfg, "FONT_PATH", None))
        self._cache: dict[int, ImageFont.FreeTypeFont] = {}
        if not self.font_path and platform.system() == "Windows":
            log.warning("Windows 下未找到 fonts/msyh.ttc，检查系统字体是否被精简")

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._cache:
            if self.font_path:
                self._cache[size] = ImageFont.truetype(self.font_path, size)
            else:
                self._cache[size] = ImageFont.load_default()
        return self._cache[size]

    # ---------------- 基础图元 ----------------

    @staticmethod
    def _bgr2rgb(c):
        return (int(c[2]), int(c[1]), int(c[0]))

    def _round_rect(self, draw, box, radius, fill=None, outline=None, width=1):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    @staticmethod
    def _truncate(draw, text: str, font, max_w: float) -> str:
        """按像素宽度截断并补省略号（中文没有空格，只能逐字二分）。"""
        if max_w <= 0 or draw.textlength(text, font=font) <= max_w:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if draw.textlength(text[:mid] + "…", font=font) <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "…"

    def draw(
        self,
        frame_bgr: np.ndarray,
        results: list,
        hud: dict | None = None,
        show_keypoints: bool = False,
    ) -> np.ndarray:
        """results: 具备 bbox/state/color/name/track_id/liveness/similarity/state_text 属性的对象列表。"""
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        h, w = frame_bgr.shape[:2]
        scale = max(0.75, min(1.4, w / 1280.0))  # 分辨率越大字号越大

        for r in results:
            color = self._bgr2rgb(r.color)
            x1, y1, x2, y2 = [int(v) for v in r.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(3 * scale)))

            if show_keypoints and getattr(r, "kps106", None) is not None:
                for (px, py) in r.kps106:
                    draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(255, 0, 180))

            # 角标：四角短线，比整框更"设备感"
            arm = int(18 * scale)
            for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                draw.line([cx, cy, cx + dx * arm, cy], fill=color, width=max(3, int(4 * scale)))
                draw.line([cx, cy, cx, cy + dy * arm], fill=color, width=max(3, int(4 * scale)))

            # ---------- 文字牌 ----------
            pad = int(8 * scale)
            f_title = self.font(int(20 * scale))
            f_info = self.font(int(15 * scale))
            name = r.name or "未知"
            line1 = f"#{r.track_id}  {name}"
            line2 = f"活体 {r.liveness:.2f}   相似度 {r.similarity:.2f}"
            line3 = r.state_text
            # 未通过时把原因也画出来（红框只说明"没过"，用户更想知道"为什么"）
            line4 = getattr(r, "reason", "") or ""
            if line4 and r.state == "passed":
                line4 = ""

            tw = max(
                draw.textlength(line1, font=f_title),
                draw.textlength(line2, font=f_info),
                draw.textlength(line3, font=f_title),
            )
            # 原因行是整块信息里最有用的，允许底板为它变宽（上限约 300px * scale），
            # 但也不能宽到占满画面
            if line4:
                reason_max = min(w - 2 * pad, max(tw + int(40 * scale), 460 * scale))
                line4 = self._truncate(draw, line4, f_info, reason_max)
                tw = max(tw, draw.textlength(line4, font=f_info))

            plate_w = int(tw) + pad * 2
            plate_h = int(26 * scale) + int(20 * scale) + int(24 * scale) + pad
            if line4:
                plate_h += int(19 * scale)
            px1 = min(max(0, x1), max(0, w - plate_w))
            py1 = y1 - plate_h - 4
            if py1 < 0:  # 上方空间不够就画在框内顶部
                py1 = y1 + 4
            px2, py2 = px1 + plate_w, py1 + plate_h

            self._round_rect(draw, [px1, py1, px2, py2], radius=int(6 * scale), fill=(15, 15, 18))
            draw.rectangle([px1, py1, px1 + int(4 * scale), py2], fill=color)

            ty = py1 + pad // 2
            draw.text((px1 + pad + int(4 * scale), ty), line1, font=f_title, fill=(245, 245, 245))
            draw.text(
                (px1 + pad + int(4 * scale), ty + int(24 * scale)),
                line2,
                font=f_info,
                fill=(190, 190, 195),
            )
            draw.text(
                (px1 + pad + int(4 * scale), ty + int(44 * scale)),
                line3,
                font=f_title,
                fill=color,
            )
            if line4:
                draw.text(
                    (px1 + pad + int(4 * scale), ty + int(68 * scale)),
                    line4,
                    font=f_info,
                    fill=(235, 205, 140),
                )

        if hud:
            self._draw_hud(draw, hud, w, h, scale)

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _draw_hud(self, draw: ImageDraw.ImageDraw, hud: dict, w: int, h: int, scale: float) -> None:
        f = self.font(int(15 * scale))
        f_big = self.font(int(22 * scale))
        items = hud.get("lines", [])
        if not items:
            return
        pad = int(10 * scale)
        line_h = int(21 * scale)
        # 宽度取"文字实测宽度"与"FPS 大字"两者的较大值，再多留一点余量，
        # 否则 msyh 等字体的 textlength 会比实际渲染窄几像素，出现文字压出底板
        text_w = max(draw.textlength(t, font=f) for t in items)
        fps_w = draw.textlength(hud.get("fps_text", ""), font=f_big)
        box_w = int(max(text_w, fps_w)) + pad * 2 + int(12 * scale)
        box_h = pad * 2 + line_h * len(items) + int(30 * scale)
        self._round_rect(draw, [pad, pad, pad + box_w, pad + box_h], radius=int(8 * scale), fill=(15, 15, 18))

        # 左侧竖条 + 大号 FPS
        fps_text = hud.get("fps_text", "")
        draw.rectangle([pad, pad, pad + int(4 * scale), pad + box_h], fill=(80, 200, 80))
        draw.text((pad + pad, pad + pad - int(2 * scale)), fps_text, font=f_big, fill=(120, 230, 120))
        y = pad + pad + int(24 * scale)
        for t in items:
            draw.text((pad + pad, y), t, font=f, fill=(200, 200, 205))
            y += line_h
