# FaceVerify

English | [中文](README_CN.md)

Real-time face verification with silent liveness detection, served as a browser-viewable web app.

The camera feed is pushed to the browser as MJPEG. Every frame runs through a full pipeline — detect, track, recognize, liveness — and is annotated with a four-level risk verdict: **Passed / Spoof / Mismatch / Unknown**.

Scope is single-person verification: only the largest face in frame is evaluated.

## Features

- `Silent liveness`: single-frame texture analysis catches printed photos, screen replay and video playback, with no user cooperation required.
- `Stable identity`: ByteTrack assigns a persistent ID, Kalman prediction smooths the box, and single-frame misses no longer produce `Unknown` flicker.
- `Two-threshold grading`: identity similarity and liveness confidence are configured independently; liveness has veto power.
- `Explainable rejection`: when liveness fails, the reason is rendered on screen (letterbox bars, face too small, underexposed, overexposed, blurred) instead of a bare score.
- `Multi-device preview`: any phone, tablet or PC on the same LAN can open the page, no plugin needed.
- `GPU with CPU fallback`: CUDA is preferred; a missing runtime degrades to CPU without interrupting the service.

## Tech Stack

| Module | Solution | Notes |
| --- | --- | --- |
| Face detection | InsightFace SCRFD `det_500m.onnx` | Bounding box, confidence and 5 keypoints in one forward pass |
| Face recognition | InsightFace ArcFace `w600k_mbf.onnx` | 512-d L2-normalized embedding, cosine similarity |
| Landmarks | InsightFace 2D106 `2d106det.onnx` | Liveness crop box and alignment inspection |
| Liveness | MiniFASNetV2 `MiniFASNetV2.onnx` | Single-frame texture analysis, 3-class output |
| Tracking | ByteTrack (lightweight port) | Kalman prediction, two-stage IoU association, ID lifecycle |
| Inference | ONNX Runtime | CUDA preferred, CPU fallback; no PyTorch/TensorFlow |
| Web | Flask | MJPEG streaming and JSON status endpoint |
| Imaging | OpenCV / NumPy / Pillow | I/O, drawing, CJK text rendering |

## Pipeline

The capture thread only reads frames into a bounded queue (oldest frame dropped when full). The inference thread consumes the newest frame and produces an annotated JPEG. The two are decoupled, so multiple viewers never fight over the camera and a slow pipeline lowers frame rate instead of accumulating latency.

| Cadence | Work |
| --- | --- |
| Every 2 frames | SCRFD detect → 2D106 landmarks → tracker update |
| Other frames | Kalman prediction only |
| New track / every 15 frames | ArcFace embedding → gallery match (with EMA fusion) |
| Every 2 frames | MiniFASNetV2 liveness |
| Every frame | Risk grading → annotation → JPEG encode |

## Risk States

| State | Box color | Condition |
| --- | --- | --- |
| Passed | Green | similarity ≥ `SIM_THRESHOLD` and liveness ≥ `LIVENESS_THRESHOLD` |
| Spoof | Red | liveness < `LIVENESS_THRESHOLD` (evaluated before identity) |
| Mismatch | Yellow | liveness passed, `MISMATCH_THRESHOLD` ≤ similarity < `SIM_THRESHOLD` |
| Unknown | White | liveness passed, similarity < `MISMATCH_THRESHOLD`, or empty gallery |

## Quick Start

### 1. Requirements

Python 3.9+ (tested on 3.13).

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Pick one dependency set:

```bash
pip install -r requirements-cpu.txt   # no NVIDIA GPU; ~26-30 FPS at 960x540 single face
pip install -r requirements-gpu.txt   # NVIDIA GPU; requires matching CUDA + cuDNN
```

The GPU build needs the CUDA runtime DLLs on `PATH`. A missing runtime is detected at startup and silently falls back to CPU. A log line `[SCRFD-det  ] backend = CUDA` (or the device badge in the UI) confirms which backend is active.

### 2. Model Files

Place these in `models/`:

| File | Purpose | Size |
| --- | --- | --- |
| `det_500m.onnx` | SCRFD detection | 2.4 MB |
| `w600k_mbf.onnx` | ArcFace embedding | 13 MB |
| `2d106det.onnx` | 2D106 landmarks | 5 MB |
| `MiniFASNetV2.onnx` | Silent liveness | 1.7 MB |

- buffalo_s package (first three): <https://github.com/deepinsight/insightface/tree/master/model_zoo>
- MiniFASNetV2: <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>

### 3. Enroll Faces

Put photos in `dataset/`; the filename is the person's name. Multiple photos per person are supported:

```
dataset/alice_1.jpg
dataset/alice_2.jpg
dataset/bob-01.jpg
```

`alice_1`, `alice-2` and `alice(3)` all resolve to `alice`. Files named like `IMG_0063.jpg` are truncated to `IMG`, so use `--name` for bulk imports of one person.

```bash
python register.py                  # parse names from filenames
python register.py --name alice     # treat every photo as the same person
python register.py --visualize      # also export aligned 112x112 crops
python register.py --clean          # rebuild from scratch
```

Source photos should be frontal, evenly lit, face width ≥ 200 px, without sunglasses or mask, yaw/pitch < 20°. HEIC is supported. Three to five photos per person is recommended.

The script prints intra-person similarity; a minimum below 0.35 usually means a profile or blurred photo slipped in.

### 4. Run

```bash
python app.py                  # 0.0.0.0:5000
python app.py --port 8000      # different port
python app.py --index 1        # different camera (built-in is usually 0)
python app.py --no-gpu         # force CPU
python app.py --no-mirror      # disable mirrored preview
```

On Windows, `start.bat` switches to the project root, prefers `.venv`, checks dependencies and starts the server.

### 5. LAN Access

Both devices must be on the same network. Open `http://<lan-ip>:5000` from the phone. If the port is blocked, allow it once:

```powershell
netsh advfirewall firewall add rule name="face-verify" dir=in action=allow protocol=TCP localport=5000
```

## Configuration

Everything lives in `config.py`; restart after editing.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `SIM_THRESHOLD` | 0.45 | Similarity required to accept an identity. Raise to 0.50-0.55 if impostors pass; lower to 0.40 if the owner is rejected |
| `MISMATCH_THRESHOLD` | 0.28 | Below this the result is `Unknown` instead of `Mismatch` |
| `LIVENESS_THRESHOLD` | 0.65 | Minimum live confidence. Lower to 0.50-0.55 for false rejections; raise to 0.75-0.85 to block more spoofs |
| `EMBEDDING_SMOOTH` | 0.5 | Weight of the newest embedding in the EMA fusion |
| `DETECT_INTERVAL` | 2 | Run detection every N frames |
| `RECOGNIZE_INTERVAL` | 15 | Recompute the embedding every N frames per track |
| `DET_SIZE` | 640 | SCRFD input size |
| `FRAME_WIDTH/HEIGHT` | 960x540 | Capture resolution |
| `SINGLE_FACE` | `True` | Evaluate only the largest face |
| `MIRROR` | `True` | Mirror the preview; analysis still runs on the original frame |

## Performance

Measured on an i7 mobile CPU (no usable CUDA), 960x540:

| Stage | Time |
| --- | --- |
| SCRFD detection | 6.6 – 11.0 ms |
| 2D106 landmarks | 3.0 – 4.0 ms |
| ArcFace embedding | 5.0 – 7.6 ms |
| MiniFASNetV2 liveness | 2.2 – 3.5 ms |
| Annotation | 2.6 – 4.5 ms |
| JPEG encode | 1.0 – 2.5 ms |
| Total | 11 ms (no face) / 27 ms (with face) |

End-to-end frame rate is bounded by camera capture, not inference. On the same machine: 1280x720 → 15.4 FPS, 960x540 → 26.3 FPS, 640x480 → 30.3 FPS. Inference alone can sustain roughly 35-90 FPS.

## Project Structure

```
.
├── app.py                Flask entry: streaming and status API
├── camera.py             Threaded capture: backend selection, MJPG, queue, reconnect
├── engine.py             Pipeline scheduling, tracking, caching, risk grading
├── config.py             All tunables
├── draw.py               CJK annotation via PIL
├── fps.py                FPS and per-stage timing
├── register.py           Batch enrollment
├── start.bat             One-click launch for Windows
├── algo/
│   ├── ort_utils.py      Session factory, CUDA probe, CPU fallback
│   ├── detector.py       SCRFD decoding and NMS
│   ├── align.py          5-point similarity transform, aligned crop
│   ├── landmark.py       2D106 landmarks
│   ├── recognizer.py     ArcFace embedding and gallery
│   ├── liveness.py       Silent liveness and rejection reasons
│   └── tracker.py        ByteTrack port
├── tools/self_test.py    Headless self-test using an image sequence
├── docs/DEVNOTES.md      Development log: problems and optimizations
├── models/               ONNX weights (not included)
├── dataset/              Enrollment photos (not included)
├── encodings/faces.pkl   Generated gallery
├── templates/ static/    Front-end
└── outputs/              Self-test output
```

## Limitations

1. Single-frame liveness cannot stop high-quality 3D masks or high-bitrate video replay; those need temporal analysis or infrared/depth hardware.
2. Scores are unreliable under very low light, severe overexposure or heavy compression.
3. Only the largest face is evaluated in single-person mode.
4. The gallery is a flat file loaded into memory; thousands of identities would need a vector index (Faiss / pgvector).
5. One process serves one camera.

## Development Notes

[docs/DEVNOTES.md](docs/DEVNOTES.md) records 16 concrete problems with symptom, root cause, fix and measured result — CUDA probing, MJPEG streaming pitfalls, liveness crop protocol, alignment template scaling, and more.

## References

- InsightFace (SCRFD / ArcFace / 2D106): <https://github.com/deepinsight/insightface>
- Silent-Face-Anti-Spoofing (MiniFASNet): <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
- ByteTrack: <https://github.com/ifzhang/ByteTrack>
- ONNX Runtime CUDA execution provider: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ArcFace, Deng et al., CVPR 2019

## License

[MIT](LICENSE)
