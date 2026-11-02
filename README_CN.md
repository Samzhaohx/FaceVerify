# FaceVerify

[English](README.md) | 中文

实时人脸核验 + 静默活体检测，以浏览器可直接访问的 Web 服务形式提供。

摄像头画面以 MJPEG 推流到浏览器；每一帧都会经过"检测 → 追踪 → 识别 → 活体"完整链路，并输出四级风控结论：**验证通过 / 活体异常 / 身份不匹配 / 未知人员**。

定位为单人核验：同一时刻只判定画面中面积最大的一张人脸。

## 功能

- `静默活体`：单帧纹理分析，拦截打印照片、屏幕翻拍与视频重放，无需用户配合动作。
- `身份稳定`：ByteTrack 分配持久 ID，卡尔曼预测平滑框体，单帧漏检不再产生 `Unknown` 跳变。
- `双阈值分级`：身份相似度与活体置信度独立配置，活体优先、一票否决。
- `拒绝可解释`：活体不通过时在画面上给出具体原因（画面有黑边、人脸偏小、偏暗、过曝、模糊），而非只显示分数。
- `多端预览`：同一局域网内的手机、平板、电脑均可打开页面，无需插件。
- `GPU/CPU 自适应`：优先使用 CUDA；运行时缺失时自动降级 CPU，服务不中断。

## 技术栈

| 模块 | 方案 | 说明 |
| --- | --- | --- |
| 人脸检测 | InsightFace SCRFD `det_500m.onnx` | 一次前向输出人脸框、置信度与 5 点关键点 |
| 人脸识别 | InsightFace ArcFace `w600k_mbf.onnx` | 512 维 L2 归一化特征，余弦相似度比对 |
| 人脸关键点 | InsightFace 2D106 `2d106det.onnx` | 活体输入裁剪与对齐质量检查 |
| 静默活体 | MiniFASNetV2 `MiniFASNetV2.onnx` | 单帧纹理分析，3 分类输出 |
| 人脸追踪 | ByteTrack（轻量实现） | 卡尔曼预测、两级 IoU 关联、ID 生命周期 |
| 推理后端 | ONNX Runtime | 优先 CUDA，自动降级 CPU；不依赖 PyTorch/TensorFlow |
| Web 服务 | Flask | MJPEG 推流与 JSON 状态接口 |
| 图像处理 | OpenCV / NumPy / Pillow | 读写、绘制、中文标注 |

## 处理流程

采集线程只负责取帧并写入有界队列（队列满时丢弃最旧帧），推理线程消费最新帧并产出带标注的 JPEG。两者解耦后，多设备同时预览不会抢占摄像头；推理变慢只会降低帧率，不会累积延迟。

| 频率 | 处理内容 |
| --- | --- |
| 每 2 帧 | SCRFD 检测 → 2D106 关键点 → 追踪更新 |
| 其余帧 | 仅做卡尔曼预测 |
| 新轨迹 / 每 15 帧 | ArcFace 特征提取 → 人脸库比对（含 EMA 融合） |
| 每 2 帧 | MiniFASNetV2 活体判定 |
| 每帧 | 风控分级 → 画面标注 → JPEG 编码 |

## 风控状态

| 状态 | 框色 | 判定条件 |
| --- | --- | --- |
| 验证通过 | 绿 | 相似度 ≥ `SIM_THRESHOLD` 且 活体 ≥ `LIVENESS_THRESHOLD` |
| 活体异常 | 红 | 活体 < `LIVENESS_THRESHOLD`（先于身份判定） |
| 身份不匹配 | 黄 | 活体通过，`MISMATCH_THRESHOLD` ≤ 相似度 < `SIM_THRESHOLD` |
| 未知人员 | 白 | 活体通过，相似度 < `MISMATCH_THRESHOLD`，或人脸库为空 |

## 快速开始

### 1. 环境

Python 3.9+（实测 3.13）。

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

依赖二选一：

```bash
pip install -r requirements-cpu.txt   # 无 NVIDIA 显卡；960x540 单脸约 26-30 FPS
pip install -r requirements-gpu.txt   # 有 NVIDIA 显卡；需匹配的 CUDA + cuDNN
```

GPU 版本需要 CUDA 运行时 DLL 位于 `PATH`。运行时缺失时，程序会在启动阶段检测到并静默降级到 CPU。日志中的 `[SCRFD-det  ] backend = CUDA`（或界面右上角的设备标识）可用于确认当前后端。

### 2. 模型文件

放入 `models/`：

| 文件 | 用途 | 大小 |
| --- | --- | --- |
| `det_500m.onnx` | SCRFD 人脸检测 | 2.4 MB |
| `w600k_mbf.onnx` | ArcFace 特征提取 | 13 MB |
| `2d106det.onnx` | 2D106 关键点 | 5 MB |
| `MiniFASNetV2.onnx` | 静默活体 | 1.7 MB |

- buffalo_s 套件（含前三个）：<https://github.com/deepinsight/insightface/tree/master/model_zoo>
- MiniFASNetV2：<https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>

### 3. 录入人脸

照片放入 `dataset/`，文件名即人名，支持一人多张：

```
dataset/alice_1.jpg
dataset/alice_2.jpg
dataset/bob-01.jpg
```

`alice_1`、`alice-2`、`alice(3)` 均解析为 `alice`。形如 `IMG_0063.jpg` 的文件名会被截断为 `IMG`，单人批量导入请使用 `--name`。

```bash
python register.py                  # 按文件名解析人名
python register.py --name alice     # 所有照片归为同一人
python register.py --visualize      # 额外导出对齐后的 112x112 人脸
python register.py --clean          # 重建
```

素材要求：正脸、光线均匀、人脸宽度 ≥ 200px、无墨镜口罩、偏航与俯仰 < 20°。支持 HEIC。建议一人 3~5 张。

脚本会打印同人相似度，最小值低于 0.35 通常意味着混入了侧脸或模糊照片。

### 4. 启动

```bash
python app.py                  # 0.0.0.0:5000
python app.py --port 8000      # 换端口
python app.py --index 1        # 换摄像头（内置一般为 0）
python app.py --no-gpu         # 强制 CPU
python app.py --no-mirror      # 关闭镜像预览
```

Windows 下可直接运行 `start.bat`：自动切换到项目根目录、优先使用 `.venv`、检查依赖后启动。

### 5. 局域网访问

两台设备需处于同一网络。手机浏览器打开 `http://<内网IP>:5000`。若端口被拦截，放行一次：

```powershell
netsh advfirewall firewall add rule name="face-verify" dir=in action=allow protocol=TCP localport=5000
```

## 配置

全部位于 `config.py`，修改后重启生效。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `SIM_THRESHOLD` | 0.45 | 判定身份匹配的相似度下限。他人误通过则提到 0.50-0.55；本人被拒则降到 0.40 |
| `MISMATCH_THRESHOLD` | 0.28 | 低于此值判为未知人员，而非身份不匹配 |
| `LIVENESS_THRESHOLD` | 0.65 | 真人置信度下限。真人被误判则降到 0.50-0.55；需拦截更多攻击则提到 0.75-0.85 |
| `EMBEDDING_SMOOTH` | 0.5 | EMA 融合中新帧特征的权重 |
| `DETECT_INTERVAL` | 2 | 每 N 帧执行一次检测 |
| `RECOGNIZE_INTERVAL` | 15 | 每条轨迹每 N 帧重算一次特征 |
| `DET_SIZE` | 640 | SCRFD 输入边长 |
| `FRAME_WIDTH/HEIGHT` | 960x540 | 采集分辨率 |
| `SINGLE_FACE` | `True` | 只判定面积最大的人脸 |
| `MIRROR` | `True` | 镜像预览，分析仍在原图上进行 |

## 性能

测试环境：i7 移动端 CPU（无可用 CUDA），960x540。

| 阶段 | 耗时 |
| --- | --- |
| SCRFD 检测 | 6.6 – 11.0 ms |
| 2D106 关键点 | 3.0 – 4.0 ms |
| ArcFace 特征 | 5.0 – 7.6 ms |
| MiniFASNetV2 活体 | 2.2 – 3.5 ms |
| 画面标注 | 2.6 – 4.5 ms |
| JPEG 编码 | 1.0 – 2.5 ms |
| 合计 | 11 ms（无人脸）/ 27 ms（有人脸） |

端到端帧率的上限在摄像头采集而非推理。同一台机器的实测：1280x720 → 15.4 FPS，960x540 → 26.3 FPS，640x480 → 30.3 FPS。纯推理能力约为 35-90 FPS。

## 目录结构

```
.
├── app.py                Flask 入口：推流与状态接口
├── camera.py             多线程采集：后端选择、MJPG、队列、断线重连
├── engine.py             流水线调度、追踪、缓存、风控分级
├── config.py             全部可调参数
├── draw.py               基于 PIL 的中文标注
├── fps.py                FPS 与分阶段耗时
├── register.py           批量录入
├── start.bat             Windows 一键启动
├── algo/
│   ├── ort_utils.py      会话工厂、CUDA 预检、CPU 降级
│   ├── detector.py       SCRFD 解码与 NMS
│   ├── align.py          5 点相似变换、对齐裁剪
│   ├── landmark.py       2D106 关键点
│   ├── recognizer.py     ArcFace 特征与特征库
│   ├── liveness.py       静默活体与拒绝原因
│   └── tracker.py        ByteTrack 实现
├── tools/self_test.py    基于图片序列的无摄像头自测
├── docs/DEVNOTES.md      开发记录：问题定位与优化
├── models/               ONNX 模型（未包含）
├── dataset/              录入照片（未包含）
├── encodings/faces.pkl   生成的人脸库
├── templates/ static/    前端页面
└── outputs/              自测输出
```

## 已知限制

1. 单帧活体无法防御高精度 3D 面具与高码率视频重放，此类场景需要时序分析或红外/深度硬件。
2. 极低照度、严重过曝或强压缩画面上，分数不可靠。
3. 单人模式下只判定面积最大的人脸。
4. 人脸库是加载进内存的平面文件；上千人规模需要向量索引（Faiss / pgvector）。
5. 一个进程服务一路摄像头。

## 开发记录

[docs/DEVNOTES.md](docs/DEVNOTES.md) 记录了 16 个具体问题，每项包含现象、根因、处理与实测结果——涵盖 CUDA 预检、MJPEG 推流、活体裁剪协议、对齐模板缩放等。

## 参考

- InsightFace（SCRFD / ArcFace / 2D106）：<https://github.com/deepinsight/insightface>
- Silent-Face-Anti-Spoofing（MiniFASNet）：<https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
- ByteTrack：<https://github.com/ifzhang/ByteTrack>
- ONNX Runtime CUDA Execution Provider：<https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ArcFace, Deng et al., CVPR 2019

## 许可

[MIT](LICENSE)
