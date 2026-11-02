// 前端状态轮询：只拉 JSON，不动视频流（MJPEG 由 <img> 自身维持连接）
(function () {
  const INTERVAL = window.__STATUS_INTERVAL__ || 500;
  const TH = window.__THRESHOLDS__ || { sim: 0.45, live: 0.65 };

  const $ = (id) => document.getElementById(id);
  const els = {
    state: $('stateText'),
    name: $('nameText'),
    track: $('trackId'),
    simVal: $('simVal'),
    liveVal: $('liveVal'),
    simBar: $('simBar'),
    liveBar: $('liveBar'),
    reason: $('reasonText'),
    diag: $('diagList'),
    card: $('resultCard'),
    device: $('devicePill'),
    fps: $('fpsPill'),
    dot: $('liveDot'),
    uptime: $('uptime'),
    kFps: $('kFps'),
    kCapFps: $('kCapFps'),
    kInfer: $('kInfer'),
    kStages: $('kStages'),
    kSize: $('kSize'),
    kDrop: $('kDrop'),
    kFrames: $('kFrames'),
    chips: $('galleryChips'),
    galleryMeta: $('galleryMeta'),
    galleryWarn: $('galleryWarn'),
    cPassed: $('cPassed'),
    cSpoof: $('cSpoof'),
    cMismatch: $('cMismatch'),
    cUnknown: $('cUnknown'),
    videoTag: $('videoTag'),
  };

  const STATE_CLASS = {
    passed: 'pass',
    spoof: 'spoof',
    mismatch: 'mismatch',
    unknown: 'unknown',
  };

  let failCount = 0;

  function setBars(sim, live, state) {
    els.simBar.style.width = Math.max(0, Math.min(100, sim * 100)) + '%';
    els.liveBar.style.width = Math.max(0, Math.min(100, live * 100)) + '%';
    // 未通过时把进度条压成对应风控色，避免"绿色进度条 + 红色结果"的视觉矛盾
    const color = {
      passed: 'linear-gradient(90deg,#2f8f4e,#50c850)',
      spoof: 'linear-gradient(90deg,#8f2f2f,#eb3c3c)',
      mismatch: 'linear-gradient(90deg,#8f7a2f,#ebc828)',
      unknown: 'linear-gradient(90deg,#3a3f47,#9aa1ab)',
    }[state] || 'linear-gradient(90deg,#3a3f47,#9aa1ab)';
    els.simBar.style.background = color;
    els.liveBar.style.background = color;
  }

  function renderDiag(face) {
    const items = (face && face.detail) || [];
    if (!items.length) {
      els.diag.hidden = true;
      els.diag.innerHTML = '';
      return;
    }
    els.diag.hidden = false;
    els.diag.className = 'diag ' + (STATE_CLASS[face.state] || 'unknown');
    els.diag.innerHTML = items
      .map((t) => '<li>' + String(t).replace(/[<>&]/g, '') + '</li>')
      .join('');
  }

  function renderFaces(d) {
    const face = (d.faces && d.faces[0]) || null;
    if (!face) {
      els.state.textContent = '等待人脸';
      els.state.className = 'state';
      els.name.textContent = '画面中没有检测到人脸';
      els.track.textContent = '#—';
      els.simVal.textContent = '—';
      els.liveVal.textContent = '—';
      els.reason.textContent = '';
      els.simBar.style.width = '0%';
      els.liveBar.style.width = '0%';
      renderDiag(null);
      return;
    }
    els.state.textContent = face.state_text;
    els.state.className = 'state ' + (STATE_CLASS[face.state] || 'unknown');
    els.track.textContent = '#' + face.track_id;
    els.name.textContent = face.name ? ('姓名 ' + face.name) : (face.state === 'spoof' ? '身份未确认（活体优先拦截）' : '身份未识别');
    els.simVal.textContent = face.similarity > 0 ? face.similarity.toFixed(3) : '—';
    els.liveVal.textContent = face.liveness.toFixed(3);
    // 灰字行：只写"分数 vs 阈值"这类客观对比；具体原因放到下面的列表里
    const tl = TH.live != null ? TH.live : 0.65;
    const ts = TH.sim != null ? TH.sim : 0.45;
    let summary = '';
    if (face.state === 'passed') summary = '身份匹配 + 活体通过';
    else if (face.state === 'spoof') summary = '活体 ' + face.liveness.toFixed(2) + ' < 阈值 ' + tl.toFixed(2);
    else if (face.state === 'mismatch') summary = '相似度 ' + face.similarity.toFixed(2) + ' < 阈值 ' + ts.toFixed(2);
    else if (face.state === 'unknown') summary = face.similarity > 0 ? '库内最相似 ' + face.similarity.toFixed(2) : '未与人脸库比对';
    els.reason.textContent = summary;
    setBars(face.similarity, face.liveness, face.state);
    renderDiag(face);
  }

  function renderMeta(d) {
    els.device.textContent = '设备 ' + (d.device || '—');
    els.fps.textContent = (d.fps || 0).toFixed(1) + ' FPS';
    els.uptime.textContent = (d.uptime || 0).toFixed(0) + ' s';
    els.kFps.textContent = (d.fps || 0).toFixed(1);
    els.kCapFps.textContent = (d.camera && d.camera.capture_fps) || 0;
    const ms = d.infer_ms || {};
    const total = Object.values(ms).reduce((a, b) => a + b, 0);
    els.kInfer.textContent = total.toFixed(1) + ' ms';
    els.kStages.textContent = [
      ms.detect != null ? 'D' + ms.detect.toFixed(0) : null,
      ms.recognize != null ? 'R' + ms.recognize.toFixed(0) : null,
      ms.liveness != null ? 'L' + ms.liveness.toFixed(0) : null,
    ].filter(Boolean).join(' / ') || '—';
    els.kSize.textContent = (d.frame_size && d.frame_size[0]) ? d.frame_size.join(' × ') : '—';
    els.kDrop.textContent = (d.camera && d.camera.dropped) || 0;
    els.kFrames.textContent = d.frames || 0;

    const c = d.counters || {};
    els.cPassed.textContent = c.passed || 0;
    els.cSpoof.textContent = c.spoof || 0;
    els.cMismatch.textContent = c.mismatch || 0;
    els.cUnknown.textContent = c.unknown || 0;

    if (d.gallery) {
      const ids = d.gallery.identities || [];
      els.galleryMeta.textContent = ids.length + ' 人 / ' + d.gallery.samples + ' 条特征';
      if (els.chips.dataset.sig !== ids.join(',')) {
        els.chips.dataset.sig = ids.join(',');
        els.chips.innerHTML = ids.length
          ? ids.map((n) => '<span class="chip">' + n.replace(/[<>&]/g, '') + '</span>').join('')
          : '<span class="muted">库为空，先运行 python register.py</span>';
      }
      // dataset/ 里有比 pkl 更新的照片 → 说明还没跑 register.py，重读库是没用的
      const pending = d.gallery.pending || 0;
      if (pending > 0) {
        els.galleryWarn.hidden = false;
        els.galleryWarn.textContent =
          '检测到 ' + pending + ' 张新照片尚未录入。请先运行 python register.py，再点「重读库」—— 只点重读不会提取特征。';
      } else {
        els.galleryWarn.hidden = true;
      }
    }
  }

  async function tick() {
    try {
      const r = await fetch('/api/status', { cache: 'no-store' });
      const d = await r.json();
      failCount = 0;
      els.dot.classList.toggle('off', !d.ok);
      if (!d.ok) {
        els.state.textContent = '引擎未就绪';
        els.state.className = 'state spoof';
        els.reason.textContent = d.error || '';
        return;
      }
      renderFaces(d);
      renderMeta(d);
    } catch (e) {
      failCount++;
      if (failCount > 3) els.dot.classList.add('off');
    }
  }

  const btn = document.getElementById('reloadBtn');
  if (btn) {
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      btn.textContent = '重载中';
      try {
        await fetch('/api/reload', { method: 'POST' });
      } catch (e) {
        /* 忽略，下一轮轮询会刷新 */
      }
      btn.disabled = false;
      btn.textContent = '重载';
      tick();
    });
  }

  tick();
  setInterval(tick, INTERVAL);
})();
