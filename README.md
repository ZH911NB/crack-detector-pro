# 路面裂缝检测系统（专业版）技术文档

> 个人知识库条目 · 项目复盘
> 关键词：`YOLO11` `Coordinate Attention(CA)` `PySide6` `QThread` `OpenCV` `PyInstaller` `目标检测` `GUI`

---

## 1. 项目概述

一个**路面裂缝目标检测**桌面应用，核心能力：

- **算法**：基于 YOLO11n 的改进模型，融入 **CA（Coordinate Attention，坐标注意力）** 机制，增强对小目标/细长裂缝特征的关注能力。
- **前端**：PySide6（Qt6）桌面 GUI，深色科技风界面，支持图片 / 视频 / 摄像头三类输入源。
- **后端**：Ultralytics YOLO 推理引擎 + OpenCV 图像/视频处理，多线程防止界面卡顿。
- **工程化**：PyInstaller 打包成免安装 exe，支持中文路径读写。

一句话架构：**「PySide6 前端交互层 + QThread 检测引擎层 + YOLO(CA) 算法推理层 + OpenCV 数据 I/O 层」**。

---

## 2. 技术栈总览

| 层次 | 技术 | 作用 |
|------|------|------|
| 算法 | YOLO11n + CA 注意力 | 裂缝检测模型 |
| 推理框架 | Ultralytics | 模型加载、前向推理、结果解析 |
| GUI | PySide6 (Qt6) | 桌面界面、信号槽、样式表 |
| 图像/视频 | OpenCV (cv2) | 读取、解码、标注、视频编码 |
| 多线程 | QThread | 检测任务与 UI 线程隔离 |
| 打包 | PyInstaller | 生成免安装 exe |

---

## 3. 系统整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (MainWindow)                     │
│  PySide6 · 深色 QSS · 信号槽 · QLabel 显示 · QTableWidget    │
│  [图片检测] [视频检测] [摄像头检测] [停止] [保存结果]          │
└──────────────┬──────────────────────────────────────────────┘
               │ Signal/Slot (frame_processed / finished / error)
┌──────────────▼──────────────────────────────────────────────┐
│                   后端检测引擎 (DetectionThread)              │
│  QThread 子线程 · OpenCV 读流 · 模型推理 · 视频编码保存        │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                    算法层 (best.pt)                          │
│            YOLO11n + Coordinate Attention (CA)               │
└─────────────────────────────────────────────────────────────┘
```

**数据流**：
输入源（图片/视频/摄像头）→ OpenCV 读帧 → YOLO 推理 → 解析 box → 绘制标注 → 回传前端显示 + 写入结果表 →（可选）编码保存视频。

---

## 4. 前端详解（PySide6）

### 4.1 界面布局

单文件 [crack_detector_pro.py](crack_detector_pro.py)，`MainWindow` 类承载全部界面。整体是**左右双栏**：

- **左侧**：两个 `QLabel` 显示「原始图像」和「检测结果」，实时刷新画面。
- **右侧**：`QScrollArea` 内嵌参数面板，包含四个 `QGroupBox`：
  1. **模型选择** —— 下拉框 + 浏览 + 加载，显示当前模型。
  2. **检测参数** —— 置信度阈值（默认 0.25）、IoU 阈值（默认 0.45），用 `QDoubleSpinBox`。
  3. **视频保存** —— 勾选框 + 保存路径（留空自动存 `results/`）。
  4. **检测操作** —— 图片/视频/摄像头/停止/保存结果五个按钮。
- 底部：`QTableWidget` 检测结果列表（类别、置信度、中心 X、中心 Y）+ 状态栏。

### 4.2 深色科技风样式（QSS）

用 `setStyleSheet` 写了一套完整的深蓝科技风主题，核心手法：

- **渐变背景**：`qlineargradient` 从 `#060e18` 到 `#0c1f35` 的深蓝渐变。
- **卡片面板**：`QFrame#main_card` / `#panel_card` 用半透明 `rgba` 渐变 + 圆角 + 边框。
- **阴影**：`QGraphicsDropShadowEffect` 给卡片加投影，增强层次感。
- **控件三态**：按钮 `:hover` / `:pressed` 不同渐变；输入框 `:focus` 高亮边框。
- **主题色**：主按钮青色系 `#1f7896`，停止按钮红色系 `#8a3d4a`，状态标签青色边框。

> 知识点：Qt 的 QSS 语法类似 CSS，支持对象选择器 `QPushButton#stop_btn`、伪状态 `:hover/:pressed/:focus`、子控件 `::title/::indicator/::down-arrow`。

### 4.3 信号槽交互

| 触发 | 槽函数 | 作用 |
|------|--------|------|
| 图片/视频/摄像头按钮 | `detect_image/video/camera` | 选文件后 `start_detection(source)` |
| 检测线程 `frame_processed` | `on_frame_processed` | 更新左右画面 + 结果表 |
| 检测线程 `finished` | `on_detection_finished` | 更新状态栏文案 |
| 检测线程 `error_occurred` | `on_detection_error` | 弹窗 + 状态提示 |
| 检测线程 `video_saved` | `on_video_saved` | 提示视频已保存 |

### 4.4 OpenCV ↔ Qt 图像转换（关键）

OpenCV 的图是 **BGR** 的 `numpy` 数组，Qt 需要 **RGB** 的 `QImage`，转换流程：

```python
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
h, w, ch = rgb.shape
qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
pixmap = QPixmap.fromImage(qimg).scaled(label.size(),
                                        Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation)
label.setPixmap(pixmap)
```

> 注意：`QImage(rgb.data, ...)` 是**零拷贝**引用 numpy 内存，`rgb` 数组在 `pixmap` 生命周期内不能被回收，否则会花屏/崩溃。当前写法在函数内局部变量会在 `setPixmap` 后失效，但因为 `QPixmap.fromImage` 已经做了深拷贝，实际安全。

---

## 5. 后端详解

### 5.1 多线程设计（QThread）

**为什么多线程**：模型推理 + 视频逐帧处理是耗时操作，若放在 UI 主线程会卡死界面。因此：

- `DetectionThread(QThread)` 子类化 `QThread`，重写 `run()`。
- 通过**自定义信号**把结果回传 UI 线程：

```python
frame_processed = Signal(np.ndarray, np.ndarray, list)  # 原始帧、标注帧、检测列表
finished        = Signal()
error_occurred  = Signal(str)
video_saved     = Signal(str)
```

- UI 侧 `self.detection_thread.frame_processed.connect(self.on_frame_processed)` 建立连接。
- `stop()` 置 `self.running = False`，配合 `quit()/wait()` 优雅停止。

> 知识点：Qt 中子线程**不能直接操作 UI**，必须通过信号槽（跨线程自动队列投递到主线程事件循环）。

### 5.2 三类输入源统一处理

`DetectionThread.run()` 里用 `is_video_source(source)` 区分：

- **视频/摄像头**（`int` 索引或视频扩展名）→ `cv2.VideoCapture` 循环读帧。
- **图片**（普通路径）→ 单次读图 + 单次推理。

视频循环里用 `self.msleep(25)` 做节流（约 40 帧上限），并在每帧后 `frame_processed.emit` 推给前端。

### 5.3 推理与结果解析

```python
results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
annotated_frame = results[0].plot()   # ultralytics 自带标注绘制
for result in results:
    for box in result.boxes:
        class_id   = int(box.cls)          # 类别 id
        class_name = self.model.names[class_id]
        confidence = float(box.conf)       # 置信度
        x, y, _, _ = box.xywh[0].tolist()  # 中心坐标
```

- `conf`/`iou` 阈值直接从界面传入，实时可调。
- `results[0].plot()` 直接生成带框的 BGR 图，省去手绘标注。

### 5.4 中文路径支持（坑点）

Windows 下 OpenCV 的 `cv2.imread` 不能读中文路径，项目用 numpy 绕开：

```python
def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)      # 二进制读入
    return cv2.imdecode(data, cv2.IMREAD_COLOR)   # 解码

def write_image_unicode(path, image):
    ok, encoded = cv2.imencode(ext, image)        # 编码
    encoded.tofile(path)                          # 二进制写出
```

### 5.5 视频编码保存（防损坏策略）

保存标注视频的流程设计得很讲究：

1. **先写临时文件**：`cv2.VideoWriter` 写到系统 `tempfile` 目录（`crack_detection_时间戳.mp4`）。
2. **编码器按扩展名选**：`.mp4` 用 `mp4v`，其它用 `XVID`。
3. **全部写完后** `shutil.move` 移动到目标路径 —— 避免中途崩溃留下损坏的视频文件。
4. `fps` 从视频读取，异常时回退到 25。

---

## 6. 算法部分（YOLO11n + CA）

### 6.1 基础模型 YOLO11n

- YOLO11 是 Ultralytics 的 YOLO 系列模型，**n（nano）** 是最轻量版本，参数量小、推理快，适合落地部署。
- 训练产物为 `best.pt`，由 `YOLO(model_path)` 一行加载即可推理。

### 6.2 CA（Coordinate Attention，坐标注意力）

在 YOLO11n 主干/颈部插入 CA 注意力模块，改进动机是**裂缝是细长、小目标**，普通卷积感受野难以精确定位：

- CA 注意力**沿高度、宽度两个空间方向**分别做全局平均池化，编码位置信息。
- 将「位置信息」与「通道注意力」融合，让网络同时关注**特征在哪个通道重要** + **特征在空间哪个位置重要**。
- 相比 SE / CBAM 等只做通道或粗粒度空间注意力，CA 对**细长条状、分布稀疏**的裂缝目标更敏感。

> 说明：本目录只保存推理端 `best.pt` 与检测 App，**训练脚本与模型结构定义不在本目录**（训练在 Ultralytics 工程下完成）。模型改进核心即「YOLO11n 网络结构中集成 CA 模块后重新训练得到的权重」。

---

## 7. 技能树（本项目覆盖的能力点）

### 前端技能
- PySide6 / Qt6 窗口编程：`QMainWindow`、布局管理（`QVBoxLayout/QHBoxLayout/QGridLayout`）。
- Qt 信号槽机制与跨线程通信。
- QSS 样式表：渐变、圆角、伪状态、阴影 `QGraphicsDropShadowEffect`。
- QLabel 动态显示图片、`QTableWidget` 展示结构化数据。

### 后端技能
- QThread 多线程 + 自定义 Signal，避免 UI 卡顿。
- OpenCV：图片/视频/摄像头读取、`imdecode/imencode` 中文路径、`VideoWriter` 视频编码。
- 临时文件 + `shutil.move` 的原子写入/防损坏策略。
- 异常处理与用户友好的错误弹窗。

### 算法技能
- YOLO 系列目标检测模型的理解与推理调用（Ultralytics API）。
- 注意力机制（CA / SE / CBAM）的原理与改进思路。
- 检测结果解析：类别、置信度、bbox（xywh/xyxy）。

### 工程化技能
- PyInstaller 打包：`Analysis/EXE` 配置、`hiddenimports`（ultralytics 的隐式导入）、`collect_all`、`datas` 打包模型权重、`console=False`、`upx=True`。
- 打包后资源路径处理：`sys.frozen` + `sys._MEIPASS` 区分开发态/打包态。
- 中文路径、跨版本兼容（cpython-37 / cpython-39 字节码共存）。

---

## 8. 项目文件结构

```
裂缝检测/
├── crack_detector_pro.py        # 主程序（前端 + 后端，单文件）
├── crack_detector_pro - 副本.py  # 旧版本备份
├── best.pt                      # YOLO11n+CA 训练权重（6.8MB）
├── crack_detector_pro.spec      # PyInstaller 打包配置（指定 best.pt 为 data）
├── 裂缝检测.spec                # 打包配置（空 hiddenimports 版本）
├── 裂缝检测系统.spec            # 打包配置（collect_all('ultralytics') 版本）
├── results/                     # 检测结果输出目录（默认保存路径）
│   └── camera_detected_*.mp4    # 摄像头检测保存的标注视频
├── build/  dist/                # PyInstaller 构建产物（已 gitignore）
└── __pycache__/                 # Python 字节码缓存
```

---

## 9. 打包部署要点

- 三个 `.spec` 文件的差异体现了**打包踩坑迭代**：
  - `crack_detector_pro.spec`：手工声明 `hiddenimports`（`ultralytics.nn.tasks` 等）+ 把 `best.pt` 打进 `datas`。
  - `裂缝检测系统.spec`：改用 `collect_all('ultralytics')` 自动收集全部隐式导入，最稳妥。
  - `裂缝检测.spec`：最简配置，作为早期版本。
- 打包产物是**单文件 exe**（`console=False` 无控制台窗口，`upx=True` 压缩）。
- 运行时用 `sys._MEIPASS` 定位解包后的模型文件：

```python
if getattr(sys, "frozen", False):
    RESOURCE_DIR = sys._MEIPASS          # 打包态：临时解包目录
    APP_DIR      = os.path.dirname(sys.executable)
else:
    RESOURCE_DIR = os.path.dirname(__file__)  # 开发态
```

---

## 10. 关键知识点与难点总结

| 难点 | 解决方案 |
|------|----------|
| 模型推理阻塞 UI | QThread 子线程 + 信号槽回传 |
| 中文路径读写 | `np.fromfile` + `cv2.imdecode` / `imencode` + `tofile` |
| BGR/RGB 颜色通道 | `cv2.cvtColor` 转 RGB 再构造 `QImage` |
| 视频保存中途崩溃产生坏文件 | 先写临时文件，完成后 `shutil.move` |
| PyInstaller 打包后 ultralytics 缺模块 | `collect_all('ultralytics')` 收集隐式导入 |
| 打包后资源路径丢失 | `sys.frozen` + `_MEIPASS` 判断运行态 |
| 模型文件多格式切换 | 统一 `YOLO()` 加载，支持 pt/onnx/engine/yaml |

---

*本文档基于项目源码 [crack_detector_pro.py](crack_detector_pro.py) 与实际打包配置整理，供个人知识库归档参考。*
