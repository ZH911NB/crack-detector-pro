# -*- coding: utf-8 -*-
"""
裂缝检测系统 - 专业版
功能：图片检测、视频检测、摄像头检测，带检测结果表格，支持保存结果
界面美观，操作简便
"""

import sys
import os
import cv2
import numpy as np
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QGridLayout, QLabel, QPushButton, QFileDialog, QTableWidget,
                               QTableWidgetItem, QHeaderView, QSpinBox, QDoubleSpinBox,
                               QGroupBox, QMessageBox, QProgressBar)
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from ultralytics import YOLO

# ========== 请修改成您自己的模型路径！ ==========
if getattr(sys, 'frozen', False):
    # 如果是打包后的 exe 运行
    base_path = sys._MEIPASS
else:
    # 如果是直接运行 .py 文件
    base_path = os.path.dirname(__file__)
MODEL_PATH = os.path.join(base_path, "best.pt")
# =============================================

class DetectionThread(QThread):
    """检测线程，避免界面卡顿"""
    frame_processed = Signal(np.ndarray, np.ndarray, list)  # 原始帧, 检测帧, 检测结果列表
    finished = Signal()

    def __init__(self, model, source, conf, iou):
        super().__init__()
        self.model = model
        self.source = source
        self.conf = conf
        self.iou = iou
        self.running = True

    def run(self):
        try:
            # 判断源类型：摄像头、视频文件、图片
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))):
                # 视频或摄像头
                cap = cv2.VideoCapture(self.source)
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # 检测
                    results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
                    annotated_frame = results[0].plot()
                    # 提取检测结果
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            class_id = int(box.cls)
                            class_name = self.model.names[class_id]
                            confidence = float(box.conf)
                            x, y, w, h = box.xywh[0].tolist()
                            detections.append((class_name, confidence, x, y))
                    # 发送信号
                    self.frame_processed.emit(frame, annotated_frame, detections)
                    # 控制帧率
                    self.msleep(30)
                cap.release()
            else:
                # 图片
                frame = cv2.imread(self.source)
                if frame is not None:
                    results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
                    annotated_frame = results[0].plot()
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            class_id = int(box.cls)
                            class_name = self.model.names[class_id]
                            confidence = float(box.conf)
                            x, y, w, h = box.xywh[0].tolist()
                            detections.append((class_name, confidence, x, y))
                    self.frame_processed.emit(frame, annotated_frame, detections)
        except Exception as e:
            print(f"检测线程错误: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("路面裂缝检测系统 - 专业版")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
  # 初始化UI
        self.init_ui()
        # 加载模型
        self.model = None
        self.load_model()

        # 线程相关
        self.detection_thread = None
        self.is_camera = False
        self.is_video = False
        self.current_result_frame = None  # 当前检测结果图，用于保存

    def init_ui(self):
        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5f9a;
            }
            QPushButton#stop_btn {
                background-color: #e74c3c;
            }
            QPushButton#stop_btn:hover {
                background-color: #c0392b;
            }
            QPushButton#save_btn {
                background-color: #27ae60;
            }
            QPushButton#save_btn:hover {
                background-color: #229954;
            }
            QLabel {
                background-color: white;
                border: 1px solid #dddddd;
                border-radius: 5px;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #dddddd;
                border-radius: 5px;
                gridline-color: #dddddd;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f0f2f5;
                padding: 5px;
                border: 1px solid #dddddd;
                font-weight: bold;
            }
            QDoubleSpinBox, QSpinBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                min-height: 20px;
            }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # ========== 左侧：图像显示区域 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 原始图像
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setScaledContents(True)
        left_layout.addWidget(self.original_label)

        # 检测结果图像
        self.result_label = QLabel("检测结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 300)
        self.result_label.setScaledContents(True)
        left_layout.addWidget(self.result_label)

        # ========== 右侧：控制面板 ==========
        right_widget = QWidget()
        right_widget.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        # 模型信息
        model_group = QGroupBox("模型信息")
        model_layout = QVBoxLayout(model_group)
        self.model_label = QLabel(f"当前模型: {os.path.basename(MODEL_PATH)}")
        model_layout.addWidget(self.model_label)
        right_layout.addWidget(model_group)

        # 检测参数
        param_group = QGroupBox("检测参数")
        param_layout = QGridLayout(param_group)

        param_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        param_layout.addWidget(self.conf_spin, 0, 1)

        param_layout.addWidget(QLabel("IoU阈值:"), 1, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        param_layout.addWidget(self.iou_spin, 1, 1)

        right_layout.addWidget(param_group)

        # 功能按钮
        btn_group = QGroupBox("功能")
        btn_layout = QVBoxLayout(btn_group)

        self.img_btn = QPushButton("图片检测")
        self.img_btn.clicked.connect(self.detect_image)
        btn_layout.addWidget(self.img_btn)

        self.video_btn = QPushButton("视频检测")
        self.video_btn.clicked.connect(self.detect_video)
        btn_layout.addWidget(self.video_btn)

        self.camera_btn = QPushButton("摄像头检测")
        self.camera_btn.clicked.connect(self.detect_camera)
        btn_layout.addWidget(self.camera_btn)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_detection)
        btn_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("保存结果")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.clicked.connect(self.save_result)
        btn_layout.addWidget(self.save_btn)

        right_layout.addWidget(btn_group)

        # 检测结果表格
        table_group = QGroupBox("检测结果列表")
        table_layout = QVBoxLayout(table_group)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["类别", "置信度", "中心X", "中心Y"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.result_table)
        right_layout.addWidget(table_group)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("background-color: #e1f5fe; padding: 5px; border-radius: 3px;")
        right_layout.addWidget(self.status_label)

        # 添加到主布局
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(right_widget, 1)

    def load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(MODEL_PATH)
            self.update_status("模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            self.update_status("模型加载失败")

    def detect_image(self):
        """图片检测"""
        if not self.check_model():
            return
        self.stop_detection()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.clear_display()
            self.is_camera = False
            self.is_video = False
            self.start_detection(file_path)

    def detect_video(self):
        """视频检测"""
        if not self.check_model():
            return
        self.stop_detection()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.clear_display()
            self.is_camera = False
            self.is_video = True
            self.start_detection(file_path)

    def detect_camera(self):
        """摄像头检测"""
        if not self.check_model():
            return
        self.stop_detection()
        self.clear_display()
        self.is_camera = True
        self.is_video = False
        self.start_detection(0)  # 默认摄像头0

    def start_detection(self, source):
        """启动检测线程"""
        conf = self.conf_spin.value()
        iou = self.iou_spin.value()
        self.detection_thread = DetectionThread(self.model, source, conf, iou)
        self.detection_thread.frame_processed.connect(self.on_frame_processed)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.start()
        self.update_status("检测中...")

    def stop_detection(self):
        """停止检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.quit()
            self.detection_thread.wait()
        self.update_status("已停止")

    def on_frame_processed(self, original_frame, result_frame, detections):
        """处理检测结果帧"""
        # 显示原始图像
        self.display_image(self.original_label, original_frame)
        # 显示检测结果图像
        self.display_image(self.result_label, result_frame)
        # 保存当前结果图
        self.current_result_frame = result_frame
        # 更新表格
        self.update_table(detections)

    def on_detection_finished(self):
        """检测结束"""
        if self.is_camera:
            self.update_status("摄像头检测已停止")
        elif self.is_video:
            self.update_status("视频检测完成")
        else:
            self.update_status("图片检测完成")

    def display_image(self, label, img):
        """在QLabel上显示OpenCV图像（BGR格式）"""
        if img is None:
            return
        # 转换为RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        # 缩放以适应label，保持宽高比
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def update_table(self, detections):
        """更新检测结果表格"""
        self.result_table.setRowCount(len(detections))
        for i, (class_name, conf, x, y) in enumerate(detections):
            self.result_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{conf:.3f}"))
            self.result_table.setItem(i, 2, QTableWidgetItem(f"{x:.1f}"))
            self.result_table.setItem(i, 3, QTableWidgetItem(f"{y:.1f}"))

    def clear_display(self):
        """清空显示区域"""
        self.original_label.clear()
        self.result_label.clear()
        self.original_label.setText("原始图像")
        self.result_label.setText("检测结果")
        self.result_table.setRowCount(0)
        self.current_result_frame = None

    def save_result(self):
        """保存当前检测结果图像"""
        if self.current_result_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的检测结果")
            return
        # 创建保存目录
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"result_{timestamp}.jpg")
        # 保存
        cv2.imwrite(save_path, self.current_result_frame)
        self.update_status(f"结果已保存至: {save_path}")

    def check_model(self):
        """检查模型是否加载成功"""
        if self.model is None:
            QMessageBox.critical(self, "错误", "模型未加载，请检查模型路径")
            return False
        return True

    def update_status(self, text):
        """更新状态栏"""
        self.status_label.setText(f"状态: {text}")

    def closeEvent(self, event):
        """关闭窗口时停止检测"""
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())