import sys
import time
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QTabWidget, QScrollArea, QGridLayout,
                            QMessageBox, QFileDialog, QComboBox, QSlider, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon
from loguru import logger
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

from core.face_detection import FaceDetector
from core.camera_manager import CameraManager
from core.alert_system import AlertEvent, AlertSystem
from core.database import FaceDatabase
from core.utils import numpy_to_pixmap, resize_image, draw_face_info
from .face_manager import FaceManagerDialog
from .alert_panel import AlertPanel
from .history_viewer import HistoryViewer

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle(f"{config['app']['name']} v{config['app']['version']}")
        self.setWindowIcon(QIcon(config['app']['logo']))
        self.setGeometry(100, 100, 1200, 800)
        
        self.processing_interval = 0.5  # seconds
        self.face_detector = FaceDetector(config)
        self.camera_manager = CameraManager('config/camera_config.yaml')
        self.alert_system = AlertSystem(config)
        self.database = FaceDatabase(config['app']['database_path'])
        
        self.face_detector.load_known_faces(config['app']['known_faces_dir'])
        
        self.init_ui()
        
        self.camera_manager.start_all_cameras()
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)  # ~30 FPS
        
        self.last_processed: Dict[int, float] = {}
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self.setup_monitor_tab()
        self.setup_controls_tab()
        self.setup_history_tab()
        
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_label)
        
        self.setup_menu_bar()
        
    def setup_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('Archivo')
        
        exit_action = file_menu.addAction('Salir')
        exit_action.triggered.connect(self.close)
        
        tools_menu = menubar.addMenu('Herramientas')
        
        face_manager_action = tools_menu.addAction('Gestor de Rostros')
        face_manager_action.triggered.connect(self.open_face_manager)
        
        alert_panel_action = tools_menu.addAction('Panel de Alertas')
        alert_panel_action.triggered.connect(self.open_alert_panel)

        view_menu = menubar.addMenu('Ver')
        
        fullscreen_action = view_menu.addAction('Pantalla Completa')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        
    def setup_monitor_tab(self):
        monitor_tab = QWidget()
        self.tab_widget.addTab(monitor_tab, "Monitor")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.camera_container = QWidget()
        self.camera_grid = QGridLayout(self.camera_container)
        self.camera_grid.setSpacing(10)
        
        scroll.setWidget(self.camera_container)
        
        layout = QVBoxLayout(monitor_tab)
        layout.addWidget(scroll)

        self.camera_labels = {}
        for cam_id in self.camera_manager.cameras:
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(400, 300)
            self.camera_labels[cam_id] = label
            self.camera_grid.addWidget(label, (cam_id // 2), (cam_id % 2))
            
    def setup_controls_tab(self):
        controls_tab = QWidget()
        self.tab_widget.addTab(controls_tab, "Controles")
        
        layout = QVBoxLayout(controls_tab)
        
        camera_group = QWidget()
        camera_layout = QVBoxLayout(camera_group)
        
        camera_title = QLabel("Controles de la Cámara")
        camera_title.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(camera_title)
        
        self.camera_combo = QComboBox()
        for cam_id, cam_config in self.camera_manager.cameras.items():
            self.camera_combo.addItem(f"Camera {cam_id}: {cam_config.name}", cam_id)
        camera_layout.addWidget(self.camera_combo)
        
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Iniciar Cámara")
        self.start_btn.clicked.connect(self.start_selected_camera)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Detener Cámara")
        self.stop_btn.clicked.connect(self.stop_selected_camera)
        btn_layout.addWidget(self.stop_btn)
        
        camera_layout.addLayout(btn_layout)
        
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Umbral de reconocimiento:")
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(int(self.config['recognition']['recognition_threshold'] * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_value = QLabel(f"{self.threshold_slider.value() / 100:.2f}")
        threshold_layout.addWidget(self.threshold_value)
        
        camera_layout.addLayout(threshold_layout)
        
        layout.addWidget(camera_group)
        
        interval_group = QWidget()
        interval_layout = QHBoxLayout(interval_group)
        
        interval_label = QLabel("Intervalo de procesamiento (ms):")
        interval_layout.addWidget(interval_label)
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 5000)
        self.interval_spin.setValue(int(self.processing_interval * 1000))
        self.interval_spin.valueChanged.connect(self.update_processing_interval)
        interval_layout.addWidget(self.interval_spin)
        
        layout.addWidget(interval_group)
        
        status_group = QWidget()
        status_layout = QVBoxLayout(status_group)
        
        status_title = QLabel("Estado del Sistema")
        status_title.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(status_title)
        
        self.status_display = QLabel("Cargando estado...")
        self.status_display.setWordWrap(True)
        status_layout.addWidget(self.status_display)
        
        layout.addWidget(status_group)
        
    def setup_history_tab(self):
        self.history_viewer = HistoryViewer(self.database, self.config)
        self.tab_widget.addTab(self.history_viewer, "Historial")
        
    def open_face_manager(self):
        dialog = FaceManagerDialog(self.face_detector, self.config['app']['known_faces_dir'])
        dialog.exec_()
        # Refresh known faces after dialog closes
        self.face_detector.load_known_faces(self.config['app']['known_faces_dir'])
        
    def open_alert_panel(self):
        dialog = AlertPanel(self.alert_system)
        dialog.exec_()
        
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def start_selected_camera(self):
        cam_id = self.camera_combo.currentData()
        if self.camera_manager.start_camera(cam_id):
            self.status_label.setText(f"Cámara iniciada {cam_id}")
            
    def stop_selected_camera(self):
        cam_id = self.camera_combo.currentData()
        if self.camera_manager.stop_camera(cam_id):
            self.status_label.setText(f"Cámara detenida {cam_id}")
            
    def update_threshold(self, value):
        threshold = value / 100
        self.face_detector.recognition_threshold = threshold
        self.threshold_value.setText(f"{threshold:.2f}")
        
    def update_processing_interval(self, value):
        self.processing_interval = value / 1000
        
    def update(self):
        try:
            frames = self.camera_manager.get_all_frames()
            
            for cam_id, frame in frames.items():
                if frame is None:
                    continue

                current_time = time.time()
                last_time = self.last_processed.get(cam_id, 0)
                if current_time - last_time < self.processing_interval:
                    self.display_frame(cam_id, frame)
                    continue

                processed_frame, alert_triggered = self.process_frame(cam_id, frame)
                
                self.display_frame(cam_id, processed_frame)
                
                self.last_processed[cam_id] = current_time
                
            self.update_status()
            
        except Exception as e:
            logger.error(f"Error en el bucle de actualización: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def process_frame(self, cam_id: int, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        alert_triggered = False
        try:
            faces = self.face_detector.detect_faces(frame)
            if not faces:
                return frame, False
                
            recognized_faces = self.face_detector.recognize_faces(faces)
            
            for face, known_face, confidence in recognized_faces:
                camera_name = self.camera_manager.cameras[cam_id].name
                
                if known_face:
                    frame = draw_face_info(
                        frame, face.bbox,
                        name=known_face.name,
                        confidence=confidence,
                        camera_name=camera_name,
                        age=face.age,
                        gender=face.gender,
                        timestamp=time.time()
                    )
                    
                    alert_event = self.alert_system.trigger_alert(
                        cam_id, camera_name,
                        known_face.name, face, confidence,
                        frame
                    )
                    alert_triggered = True

                    self.database.log_face_event(alert_event)
                else:
                    frame = draw_face_info(
                        frame, face.bbox,
                        name="Unknown",
                        confidence=confidence,
                        camera_name=camera_name,
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            
        return frame, alert_triggered
        
    def display_frame(self, cam_id: int, frame: np.ndarray):
        try:
            if frame is None:
                return
                
            pixmap = numpy_to_pixmap(frame)
            self.camera_labels[cam_id].setPixmap(pixmap)
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
            
    def update_status(self):
        try:
            status_text = []
            
            status_text.append("=== Estado de la cámara ===")
            for cam_id, cam_config in self.camera_manager.cameras.items():
                running = cam_id in self.camera_manager.capture_threads
                status_text.append(
                    f"Camera {cam_id} ({cam_config.name}): {'En ejecución' if running else 'Detenida'}"
                )
                
            status_text.append("\n=== Base de datos de rostros ===")
            status_text.append(f"Rostros conocidos: {len(self.face_detector.known_faces)}")
            
            status_text.append("\n=== Alerts ===")
            recent_alerts = self.alert_system.get_recent_alerts(3)
            if recent_alerts:
                for alert in recent_alerts:
                    time_str = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
                    status_text.append(
                        f"{time_str}: {alert.face_name} on {alert.camera_name} "
                        f"(Confianza: {alert.confidence:.2f})"
                    )
            else:
                status_text.append("Sin alertas recientes")
                
            self.status_display.setText("\n".join(status_text))
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            
    def closeEvent(self, event):
        try:
            self.camera_manager.stop_all_cameras()
            
            self.update_timer.stop()
            
            event.accept()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            event.accept()