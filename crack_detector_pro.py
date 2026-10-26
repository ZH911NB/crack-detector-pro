# -*- coding: utf-8 -*-
"""
裂缝检测系统 - 专业版
功能：图片检测、视频检测、摄像头检测、模型切换、结果图片保存、检测视频保存。
"""

import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QAbstractItemView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


if getattr(sys, "frozen", False):
    RESOURCE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    RESOURCE_DIR = os.path.dirname(__file__)
    APP_DIR = RESOURCE_DIR

DEFAULT_MODEL_PATH = os.path.join(RESOURCE_DIR, "best.pt")
DEFAULT_SAVE_DIR = os.path.join(APP_DIR, "results")
MODEL_EXTENSIONS = (".pt", ".onnx", ".engine", ".yaml", ".yml")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def read_image_unicode(path):
    """支持中文路径的 OpenCV 图片读取。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(path, image):
    """支持中文路径的 OpenCV 图片保存。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if ok:
        encoded.tofile(path)
    return ok


def is_video_source(source):
    return isinstance(source, int) or (
        isinstance(source, str) and source.lower().endswith(VIDEO_EXTENSIONS)
    )


class DetectionThread(QThread):
    """检测线程，避免界面卡顿。"""

    frame_processed = Signal(np.ndarray, np.ndarray, list)
    finished = Signal()
    error_occurred = Signal(str)
    video_saved = Signal(str)

    def __init__(self, model, source, conf, iou, save_video_path=None):
        super().__init__()
        self.model = model
        self.source = source
        self.conf = conf
        self.iou = iou
        self.save_video_path = save_video_path
        self.running = True
        self._temp_video_path = None
        self._writer_failed = False

    def run(self):
        writer = None
        cap = None
        saved_path = None

        try:
            if is_video_source(self.source):
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    self.error_occurred.emit("无法打开视频或摄像头")
                    return

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 1 or np.isnan(fps):
                    fps = 25

                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated_frame, detections = self.detect_frame(frame)

                    if self.save_video_path and not self._writer_failed:
                        if writer is None:
                            writer = self.create_video_writer(
                                self.save_video_path,
                                annotated_frame.shape[1],
                                annotated_frame.shape[0],
                                fps,
                            )
                        if writer is not None:
                            writer.write(annotated_frame)

                    self.frame_processed.emit(frame, annotated_frame, detections)
                    self.msleep(25)

                if writer is not None:
                    writer.release()
                    writer = None
                    saved_path = self.finalize_video_file()

            else:
                frame = read_image_unicode(self.source)
                if frame is None:
                    self.error_occurred.emit("无法读取图片文件")
                    return

                annotated_frame, detections = self.detect_frame(frame)
                self.frame_processed.emit(frame, annotated_frame, detections)

        except Exception as exc:
            self.error_occurred.emit(f"检测线程错误: {exc}")
        finally:
            if writer is not None:
                writer.release()
                if self.save_video_path:
                    saved_path = self.finalize_video_file()
            if cap is not None:
                cap.release()
            if saved_path:
                self.video_saved.emit(saved_path)
            self.finished.emit()

    def detect_frame(self, frame):
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        annotated_frame = results[0].plot()
        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                confidence = float(box.conf)
                x, y, _, _ = box.xywh[0].tolist()
                detections.append((class_name, confidence, x, y))

        return annotated_frame, detections

    def create_video_writer(self, output_path, width, height, fps):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower() or ".mp4"
        fourcc_name = "mp4v" if ext == ".mp4" else "XVID"

        temp_name = f"crack_detection_{int(time.time())}{ext}"
        self._temp_video_path = os.path.join(tempfile.gettempdir(), temp_name)
        writer = cv2.VideoWriter(
            self._temp_video_path,
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            self.error_occurred.emit("检测视频保存失败：无法创建视频文件")
            self._writer_failed = True
            return None
        return writer

    def finalize_video_file(self):
        if not self._temp_video_path or not self.save_video_path:
            return None
        if not os.path.exists(self._temp_video_path):
            return None

        os.makedirs(os.path.dirname(self.save_video_path) or ".", exist_ok=True)
        if os.path.exists(self.save_video_path):
            os.remove(self.save_video_path)
        shutil.move(self._temp_video_path, self.save_video_path)
        return self.save_video_path

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("路面裂缝检测系统 - 专业版")
        self.setGeometry(100, 100, 1280, 820)
        self.setMinimumSize(1080, 720)

        self.model = None
        self.current_model_path = DEFAULT_MODEL_PATH
        self.detection_thread = None
        self.is_camera = False
        self.is_video = False
        self.current_result_frame = None
        self.last_video_save_path = None

        self.init_ui()
        self.refresh_model_list()
        self.load_model(self.current_model_path)

    def init_ui(self):
        self.setStyleSheet("""
            /* ========== 全局基底 ========== */
            QMainWindow {
                background: #060e18;
            }
            QWidget#bg_root {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #060e1a, stop:0.3 #091a2c,
                                            stop:0.7 #0c1f35, stop:1 #0b1728);
            }
            QWidget {
                background: transparent;
                color: #e2edf9;
                font-family: "Microsoft YaHei UI", "Segoe UI";
            }

            /* ========== 顶部标题栏 ========== */
            QFrame#header_bar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #082030, stop:0.25 #0f3148,
                                            stop:0.5 #13405a, stop:0.75 #0f3148,
                                            stop:1 #081e2e);
                border: 1px solid #2e7a9e;
                border-radius: 12px;
            }
            QLabel#title_label {
                color: #f0f8ff;
                font-size: 38px;
                font-weight: 900;
            }
            QLabel#subtitle_label {
                color: #87ceeb;
                font-size: 13px;
                letter-spacing: 1px;
            }

            /* ========== 卡片面板 ========== */
            QFrame#main_card {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 rgba(10,24,40,240), stop:1 rgba(14,30,48,235));
                border: 1px solid #2a5c78;
                border-radius: 12px;
            }
            QFrame#panel_card {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 rgba(12,26,42,242), stop:1 rgba(8,20,32,238));
                border: 1px solid #2a5570;
                border-radius: 12px;
            }

            /* ========== 分组框 ========== */
            QGroupBox {
                font-weight: 700;
                font-size: 13px;
                border: 1px solid #2a5f7a;
                border-radius: 10px;
                margin-top: 13px;
                padding: 16px 10px 10px 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 rgba(14,32,48,220), stop:1 rgba(10,22,34,210));
                color: #d0e8ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 2px 10px;
                color: #b8e2ff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #13374d, stop:1 #1a4a62);
                border: 1px solid #2a7090;
                border-radius: 5px;
            }

            /* ========== 标签 ========== */
            QLabel {
                background: transparent;
                border: none;
                color: #d0e5f8;
                font-size: 13px;
            }
            QLabel#image_panel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #040a12, stop:1 #071019);
                border: 2px solid #2d6078;
                border-radius: 10px;
                color: #7a9db5;
                font-size: 17px;
            }
            QLabel#status_label {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #0a1a2a, stop:1 #0e2236);
                border: 1px solid #2a6d8a;
                color: #e0f0ff;
                padding: 10px;
                border-radius: 8px;
                font-size: 13px;
            }

            /* ========== 按钮 - 主按钮渐变 ========== */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #1f7896, stop:0.5 #1a6882, stop:1 #155a72);
                color: #f4fbff;
                border: 1px solid #4da5c4;
                padding: 7px 14px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
                min-height: 28px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #2a8fb0, stop:0.5 #2280a0, stop:1 #1a6e8c);
                border-color: #7ad4f0;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #155a72, stop:1 #1a6882);
            }
            QPushButton#stop_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #8a3d4a, stop:0.5 #7a3340, stop:1 #682b36);
                border-color: #c06070;
            }
            QPushButton#stop_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #a54959, stop:0.5 #943f4e, stop:1 #7e3542);
                border-color: #e08090;
            }
            QPushButton#secondary_btn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #243a4e, stop:0.5 #1e3244, stop:1 #182a3a);
                border-color: #50738a;
            }
            QPushButton#secondary_btn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #2e485e, stop:0.5 #284054, stop:1 #22384a);
                border-color: #7098b2;
            }

            /* ========== 输入控件 ========== */
            QLineEdit, QComboBox, QDoubleSpinBox {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #06111d, stop:1 #091826);
                border: 1px solid #2d5872;
                border-radius: 7px;
                color: #e6f2ff;
                min-height: 30px;
                padding: 4px 10px;
                selection-background-color: #1e7a9a;
            }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
                border-color: #5db8d8;
            }
            QComboBox::drop-down {
                border: none;
                width: 26px;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }
            QComboBox::down-arrow {
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background: #091826;
                border: 1px solid #2d5872;
                border-radius: 6px;
                color: #e6f2ff;
                selection-background-color: #1e7a9a;
                outline: none;
            }

            /* ========== 复选框 ========== */
            QCheckBox {
                color: #cde4f8;
                spacing: 10px;
                font-size: 13px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #4d809a;
                border-radius: 5px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #07131f, stop:1 #0b1d2e);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #2598b8, stop:1 #1c7e9a);
                border-color: #7ad4ee;
            }

            /* ========== 表格 ========== */
            QTableWidget {
                background: #07131f;
                border: 1px solid #2d5872;
                border-radius: 8px;
                gridline-color: #1e3448;
                color: #e0f0ff;
                selection-background-color: #1e6a85;
                alternate-background-color: #0a1a2a;
            }
            QTableWidget::item {
                padding: 7px;
                border-bottom: 1px solid #14283a;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #15364c, stop:1 #0f2a3c);
                padding: 8px;
                border: 1px solid #2d5872;
                font-weight: 700;
                color: #d8f0ff;
            }

            /* ========== 滚动条 ========== */
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #081420;
                border-radius: 5px;
            }
            QScrollBar:vertical {
                width: 10px;
            }
            QScrollBar:horizontal {
                height: 10px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #2a5a70, stop:1 #356d85);
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #357090, stop:1 #4088a2);
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
            QScrollArea#side_scroll {
                background: transparent;
                border: none;
            }
            QWidget#side_content {
                background: transparent;
            }

            /* ========== 提示文本 ========== */
            QToolTip {
                background: #0d2136;
                color: #e0f0ff;
                border: 1px solid #3d708a;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 12px;
            }
        """)

        central_widget = QWidget()
        central_widget.setObjectName("bg_root")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(14, 12, 14, 14)

        top_header = QFrame()
        top_header.setObjectName("header_bar")
        top_header_layout = QVBoxLayout(top_header)
        top_header_layout.setSpacing(2)
        top_header_layout.setContentsMargins(10, 10, 10, 10)

        title_label = QLabel("路面裂缝检测系统")
        title_label.setObjectName("title_label")
        title_label.setAlignment(Qt.AlignCenter)
        subtitle_label = QLabel("Crack Detection Pro  |  图片、视频、摄像头实时检测")
        subtitle_label.setObjectName("subtitle_label")
        subtitle_label.setAlignment(Qt.AlignCenter)

        top_header_layout.addWidget(title_label)
        top_header_layout.addWidget(subtitle_label)

        # 装饰性科技分割线
        accent_line = QFrame()
        accent_line.setFixedHeight(2)
        accent_line.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 transparent, stop:0.2 #1a6a88,
                                            stop:0.5 #3db8dc, stop:0.8 #1a6a88,
                                            stop:1 transparent);
                border: none;
                border-radius: 1px;
            }
        """)
        top_header_layout.addWidget(accent_line)
        root_layout.addWidget(top_header)

        main_card = QFrame()
        main_card.setObjectName("main_card")
        main_layout = QHBoxLayout(main_card)
        main_layout.setSpacing(14)
        main_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.addWidget(main_card, 1)

        left_widget = QFrame()
        left_widget.setObjectName("panel_card")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(12)

        self.original_label = QLabel("原始图像")
        self.original_label.setObjectName("image_panel")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(320, 240)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.original_label, 1)

        self.result_label = QLabel("检测结果")
        self.result_label.setObjectName("image_panel")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(320, 240)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.result_label, 1)

        right_widget = QFrame()
        right_widget.setObjectName("panel_card")
        right_widget.setMinimumWidth(330)
        right_widget.setMaximumWidth(520)
        right_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_shell_layout = QVBoxLayout(right_widget)
        right_shell_layout.setContentsMargins(4, 8, 4, 8)

        side_scroll = QScrollArea()
        side_scroll.setObjectName("side_scroll")
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        side_content = QWidget()
        side_content.setObjectName("side_content")
        right_layout = QVBoxLayout(side_content)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(8, 6, 8, 6)
        side_scroll.setWidget(side_content)
        right_shell_layout.addWidget(side_scroll)

        model_group = QGroupBox("模型选择")
        model_layout = QGridLayout(model_group)
        model_layout.setHorizontalSpacing(8)
        model_layout.setVerticalSpacing(9)

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_combo_changed)
        model_layout.addWidget(self.model_combo, 0, 0, 1, 2)

        self.browse_model_btn = QPushButton("浏览模型")
        self.browse_model_btn.setObjectName("secondary_btn")
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn, 1, 0)

        self.reload_model_btn = QPushButton("加载模型")
        self.reload_model_btn.clicked.connect(self.reload_selected_model)
        model_layout.addWidget(self.reload_model_btn, 1, 1)

        self.model_label = QLabel("当前模型: 未加载")
        model_layout.addWidget(self.model_label, 2, 0, 1, 2)
        right_layout.addWidget(model_group)

        param_group = QGroupBox("检测参数")
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(10)

        conf_label = QLabel("置信度阈值")
        param_layout.addWidget(conf_label, 0, 0)
        self.conf_spin = self.create_double_spin(0.01, 1.0, 0.25, 0.05)
        param_layout.addWidget(self.conf_spin, 0, 1)

        iou_label = QLabel("IoU 阈值")
        param_layout.addWidget(iou_label, 1, 0)
        self.iou_spin = self.create_double_spin(0.01, 1.0, 0.45, 0.05)
        param_layout.addWidget(self.iou_spin, 1, 1)

        right_layout.addWidget(param_group)

        save_group = QGroupBox("视频保存")
        save_layout = QGridLayout(save_group)
        save_layout.setHorizontalSpacing(8)
        save_layout.setVerticalSpacing(9)

        self.save_video_check = QCheckBox("视频检测时保存标注视频")
        self.save_video_check.setChecked(True)
        save_layout.addWidget(self.save_video_check, 0, 0, 1, 2)

        self.video_save_path_edit = QLineEdit()
        self.video_save_path_edit.setPlaceholderText("留空则自动保存到 results 文件夹")
        save_layout.addWidget(self.video_save_path_edit, 1, 0)

        self.browse_video_save_btn = QPushButton("选择")
        self.browse_video_save_btn.setObjectName("secondary_btn")
        self.browse_video_save_btn.clicked.connect(self.browse_video_save_path)
        save_layout.addWidget(self.browse_video_save_btn, 1, 1)

        right_layout.addWidget(save_group)

        btn_group = QGroupBox("检测操作")
        btn_layout = QGridLayout(btn_group)
        btn_layout.setContentsMargins(10, 14, 10, 10)
        btn_layout.setHorizontalSpacing(8)
        btn_layout.setVerticalSpacing(8)
        btn_layout.setColumnStretch(0, 1)
        btn_layout.setColumnStretch(1, 1)

        self.img_btn = QPushButton("图片检测")
        self.img_btn.clicked.connect(self.detect_image)
        btn_layout.addWidget(self.img_btn, 0, 0)

        self.video_btn = QPushButton("视频检测")
        self.video_btn.clicked.connect(self.detect_video)
        btn_layout.addWidget(self.video_btn, 0, 1)

        self.camera_btn = QPushButton("摄像头检测")
        self.camera_btn.clicked.connect(self.detect_camera)
        btn_layout.addWidget(self.camera_btn, 1, 0)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_detection)
        btn_layout.addWidget(self.stop_btn, 1, 1)

        self.save_btn = QPushButton("保存当前结果图")
        self.save_btn.clicked.connect(self.save_result)
        btn_layout.addWidget(self.save_btn, 2, 0, 1, 2)

        self.action_buttons = [
            self.img_btn,
            self.video_btn,
            self.camera_btn,
            self.stop_btn,
            self.save_btn,
        ]
        for btn in self.action_buttons:
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        for btn in [
            self.browse_model_btn,
            self.reload_model_btn,
            self.browse_video_save_btn,
        ]:
            btn.setMinimumHeight(34)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        right_layout.addWidget(btn_group)

        table_group = QGroupBox("检测结果列表")
        table_layout = QVBoxLayout(table_group)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["类别", "置信度", "中心 X", "中心 Y"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setMinimumHeight(150)
        table_layout.addWidget(self.result_table)
        right_layout.addWidget(table_group)

        self.status_label = QLabel("状态: 就绪")
        self.status_label.setObjectName("status_label")
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        right_font = QFont("Microsoft YaHei UI", 11)
        for widget in [
            self.model_combo,
            self.browse_model_btn,
            self.reload_model_btn,
            self.model_label,
            conf_label,
            iou_label,
            self.conf_spin,
            self.iou_spin,
            self.save_video_check,
            self.video_save_path_edit,
            self.browse_video_save_btn,
            self.img_btn,
            self.video_btn,
            self.camera_btn,
            self.stop_btn,
            self.save_btn,
            self.result_table,
            self.status_label,
        ]:
            widget.setFont(right_font)

        self.right_widget = right_widget
        self.side_scroll = side_scroll
        self.btn_layout = btn_layout

        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(right_widget, 1)

        self._add_shadow(main_card, blur=40, offset_y=3, color=(0, 0, 0, 140))
        self._add_shadow(top_header, blur=30, offset_y=2, color=(20, 130, 170, 100))
        self._add_shadow(left_widget, blur=30, offset_y=2, color=(0, 30, 50, 110))
        self._add_shadow(right_widget, blur=30, offset_y=2, color=(0, 30, 50, 110))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "action_buttons"):
            return

        compact = self.height() < 820
        button_height = 34 if compact else 40
        spacing = 6 if compact else 8
        self.btn_layout.setHorizontalSpacing(spacing)
        self.btn_layout.setVerticalSpacing(spacing)
        for button in self.action_buttons:
            button.setMinimumHeight(button_height)
            button.setMaximumHeight(button_height)

    def create_double_spin(self, minimum, maximum, value, step):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _add_shadow(self, widget, blur=20, offset_y=2, color=(0, 0, 0, 120)):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, offset_y)
        shadow.setColor(QColor(*color))
        widget.setGraphicsEffect(shadow)

    def refresh_model_list(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        model_paths = []
        for folder in [RESOURCE_DIR, APP_DIR]:
            if os.path.isdir(folder):
                for name in os.listdir(folder):
                    path = os.path.join(folder, name)
                    if os.path.isfile(path) and name.lower().endswith(MODEL_EXTENSIONS):
                        model_paths.append(path)

        if os.path.exists(DEFAULT_MODEL_PATH):
            model_paths.insert(0, DEFAULT_MODEL_PATH)

        unique_paths = []
        for path in model_paths:
            normalized = os.path.normcase(os.path.abspath(path))
            if normalized not in [os.path.normcase(os.path.abspath(p)) for p in unique_paths]:
                unique_paths.append(path)

        for path in unique_paths:
            self.model_combo.addItem(os.path.basename(path), path)

        if unique_paths:
            selected_index = 0
            for index, path in enumerate(unique_paths):
                if os.path.abspath(path) == os.path.abspath(self.current_model_path):
                    selected_index = index
                    break
            self.model_combo.setCurrentIndex(selected_index)
            self.current_model_path = self.model_combo.currentData()

        self.model_combo.blockSignals(False)

    def on_model_combo_changed(self, _index=None):
        path = self.model_combo.currentData()
        if path:
            self.current_model_path = path
            self.model_label.setText(f"待加载模型: {os.path.basename(path)}")

    def browse_model(self):
        self.stop_detection()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            APP_DIR,
            "模型文件 (*.pt *.onnx *.engine *.yaml *.yml);;所有文件 (*.*)",
        )
        if not file_path:
            return

        self.current_model_path = file_path
        index = self.model_combo.findData(file_path)
        if index < 0:
            self.model_combo.addItem(os.path.basename(file_path), file_path)
            index = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(index)
        self.load_model(file_path)

    def reload_selected_model(self):
        path = self.model_combo.currentData() or self.current_model_path
        if not path:
            QMessageBox.warning(self, "提示", "请先选择模型文件")
            return
        self.stop_detection()
        self.load_model(path)

    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.current_model_path = model_path
            self.model_label.setText(f"当前模型: {os.path.basename(model_path)}")
            self.update_status("模型加载成功")
        except Exception as exc:
            self.model = None
            self.model_label.setText("当前模型: 加载失败")
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{exc}")
            self.update_status("模型加载失败")

    def detect_image(self):
        if not self.check_model():
            return
        self.stop_detection()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            APP_DIR,
            "图片文件 (*.jpg *.jpeg *.png *.bmp)",
        )
        if file_path:
            self.clear_display()
            self.is_camera = False
            self.is_video = False
            self.start_detection(file_path)

    def detect_video(self):
        if not self.check_model():
            return
        self.stop_detection()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            APP_DIR,
            "视频文件 (*.mp4 *.avi *.mov *.mkv)",
        )
        if file_path:
            self.clear_display()
            self.is_camera = False
            self.is_video = True
            self.start_detection(file_path)

    def detect_camera(self):
        if not self.check_model():
            return
        self.stop_detection()
        self.clear_display()
        self.is_camera = True
        self.is_video = False
        self.start_detection(0)

    def start_detection(self, source):
        save_video_path = self.resolve_video_save_path(source)
        conf = self.conf_spin.value()
        iou = self.iou_spin.value()

        self.last_video_save_path = save_video_path
        self.detection_thread = DetectionThread(
            self.model,
            source,
            conf,
            iou,
            save_video_path=save_video_path,
        )
        self.detection_thread.frame_processed.connect(self.on_frame_processed)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error_occurred.connect(self.on_detection_error)
        self.detection_thread.video_saved.connect(self.on_video_saved)
        self.detection_thread.start()

        if save_video_path:
            self.update_status("检测中，正在保存标注视频...")
        else:
            self.update_status("检测中...")

    def resolve_video_save_path(self, source):
        if not is_video_source(source) or not self.save_video_check.isChecked():
            return None

        custom_path = self.video_save_path_edit.text().strip()
        if custom_path:
            root, ext = os.path.splitext(custom_path)
            if ext.lower() not in VIDEO_EXTENSIONS:
                custom_path = f"{root}.mp4"
            return custom_path

        os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = "camera" if isinstance(source, int) else os.path.splitext(os.path.basename(source))[0]
        return os.path.join(DEFAULT_SAVE_DIR, f"{prefix}_detected_{timestamp}.mp4")

    def browse_video_save_path(self):
        os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
        default_path = os.path.join(
            DEFAULT_SAVE_DIR,
            f"detected_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测视频",
            default_path,
            "MP4 视频 (*.mp4);;AVI 视频 (*.avi)",
        )
        if file_path:
            self.video_save_path_edit.setText(file_path)
            self.save_video_check.setChecked(True)

    def stop_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.quit()
            self.detection_thread.wait()
        self.update_status("已停止")

    def on_frame_processed(self, original_frame, result_frame, detections):
        self.display_image(self.original_label, original_frame)
        self.display_image(self.result_label, result_frame)
        self.current_result_frame = result_frame
        self.update_table(detections)

    def on_detection_finished(self):
        if self.is_camera:
            self.update_status("摄像头检测已停止")
        elif self.is_video:
            if self.last_video_save_path:
                self.update_status(f"视频检测完成，已保存至: {self.last_video_save_path}")
            else:
                self.update_status("视频检测完成")
        else:
            self.update_status("图片检测完成")

    def on_detection_error(self, message):
        if "检测视频保存失败" in message:
            self.last_video_save_path = None
        self.update_status(message)
        QMessageBox.warning(self, "检测提示", message)

    def on_video_saved(self, path):
        self.last_video_save_path = path
        self.update_status(f"检测视频已保存至: {path}")

    def display_image(self, label, img):
        if img is None:
            return

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def update_table(self, detections):
        self.result_table.setRowCount(len(detections))
        for row, (class_name, conf, x, y) in enumerate(detections):
            self.result_table.setItem(row, 0, QTableWidgetItem(str(class_name)))
            self.result_table.setItem(row, 1, QTableWidgetItem(f"{conf:.3f}"))
            self.result_table.setItem(row, 2, QTableWidgetItem(f"{x:.1f}"))
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{y:.1f}"))

    def clear_display(self):
        self.original_label.clear()
        self.result_label.clear()
        self.original_label.setText("原始图像")
        self.result_label.setText("检测结果")
        self.result_table.setRowCount(0)
        self.current_result_frame = None

    def save_result(self):
        if self.current_result_frame is None:
            QMessageBox.warning(self, "提示", "没有可保存的检测结果")
            return

        os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(DEFAULT_SAVE_DIR, f"result_{timestamp}.jpg")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前结果图",
            default_path,
            "JPEG 图片 (*.jpg);;PNG 图片 (*.png);;BMP 图片 (*.bmp)",
        )
        if not save_path:
            return

        if write_image_unicode(save_path, self.current_result_frame):
            self.update_status(f"结果图已保存至: {save_path}")
        else:
            QMessageBox.warning(self, "错误", "结果图保存失败")

    def check_model(self):
        if self.model is None:
            QMessageBox.critical(self, "错误", "模型未加载，请先选择并加载模型")
            return False
        return True

    def update_status(self, text):
        self.status_label.setText(f"状态: {text}")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
