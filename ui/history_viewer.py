from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
                            QLabel, QDateEdit, QComboBox, QSpacerItem, QSizePolicy,
                            QSplitter, QFrame, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QDate, QDateTime
from PyQt5.QtGui import QPixmap
from loguru import logger
import time
from datetime import datetime, timedelta
from typing import List, Optional
import cv2

from core.database import FaceDatabase, FaceLogEntry
from core.utils import numpy_to_pixmap

class HistoryViewer(QWidget):
    def __init__(self, database, config):
        super().__init__()
        self.database = database
        self.config = config
        self.current_entry = None
        
        self.setup_ui()
        self.load_camera_list()
        self.load_face_list()
        self.refresh_history()
        
    def setup_ui(self):
        main_layout = QVBoxLayout()
        
        filter_layout = QHBoxLayout()
        
        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Rango de Fechas:"))
        
        date_range_layout = QHBoxLayout()
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.start_date.setCalendarPopup(True)
        date_range_layout.addWidget(self.start_date)
        
        date_range_layout.addWidget(QLabel("hasta"))
        
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        date_range_layout.addWidget(self.end_date)
        
        date_layout.addLayout(date_range_layout)
        filter_layout.addLayout(date_layout)
        
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(QLabel("Cámara:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Todas las Cámaras", None)
        camera_layout.addWidget(self.camera_combo)
        filter_layout.addLayout(camera_layout)
        
        face_layout = QVBoxLayout()
        face_layout.addWidget(QLabel("Rostro:"))
        self.face_combo = QComboBox()
        self.face_combo.addItem("Todos los Rostros", None)
        face_layout.addWidget(self.face_combo)
        filter_layout.addLayout(face_layout)
        
        self.refresh_btn = QPushButton("Refrescar")
        self.refresh_btn.clicked.connect(self.refresh_history)
        filter_layout.addWidget(self.refresh_btn)
        
        filter_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(filter_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        
        self.history_list = QListWidget()
        self.history_list.currentItemChanged.connect(self.on_history_item_selected)
        splitter.addWidget(self.history_list)
        
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.StyledPanel)
        details_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        details_layout.addWidget(self.image_label)
        
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        details_layout.addWidget(self.details_label)
        
        self.view_screenshot_btn = QPushButton("Ver Captura de Pantalla")
        self.view_screenshot_btn.clicked.connect(self.view_screenshot)
        details_layout.addWidget(self.view_screenshot_btn)
        
        details_frame.setLayout(details_layout)
        splitter.addWidget(details_frame)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)
        
    def load_camera_list(self):
        try:
            with open('config/camera_config.yaml', 'r') as f:
                import yaml
                config = yaml.safe_load(f)
                
            for camera in config.get('cameras', []):
                self.camera_combo.addItem(
                    f"Camera {camera['id']}: {camera.get('name', '')}",
                    camera['id']
                )
                
        except Exception as e:
            logger.error(f"Error al cargar la configuración de la cámara: {e}")
            
    def load_face_list(self):
        try:
            known_faces = self.database.get_known_faces()
            for face in known_faces:
                self.face_combo.addItem(face['name'], face['name'])
                
        except Exception as e:
            logger.error(f"Error al cargar caras conocidas: {e}")

    def refresh_history(self):
        try:
            start_date = self.start_date.date().toPyDate()
            end_date = self.end_date.date().toPyDate() + timedelta(days=1)
            
            start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
            end_timestamp = datetime.combine(end_date, datetime.min.time()).timestamp()
            
            camera_id = self.camera_combo.currentData()
            face_name = self.face_combo.currentData()
            
            entries = self.database.get_face_logs(
                limit=1000,
                camera_id=camera_id,
                face_name=face_name,
                start_time=start_timestamp,
                end_time=end_timestamp
            )
            
            self.history_list.clear()
            for entry in entries:
                try:
                    timestamp = float(entry.timestamp) if isinstance(entry.timestamp, bytes) else entry.timestamp
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    item_text = f"{time_str} - {entry.face_name} on {entry.camera_name}"
                    self.history_list.addItem(item_text)
                    self.history_list.item(self.history_list.count() - 1).setData(Qt.UserRole, entry)
                except Exception as e:
                    logger.error(f"Error al procesar la entrada del historial: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error al actualizar el historial: {e}")

    def on_history_item_selected(self, current, previous):
        try:
            if current is None:
                self.current_entry = None
                self.image_label.clear()
                self.details_label.clear()
                self.view_screenshot_btn.setEnabled(False)
                return
                
            entry = current.data(Qt.UserRole)
            if not isinstance(entry, FaceLogEntry):
                return
                
            self.current_entry = entry
            
            try:
                timestamp = float(entry.timestamp)
                time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError) as e:
                logger.warning(f"Formato de marca de tiempo no válido: {entry.timestamp}")
                time_str = "Hora desconocida"
            
            try:
                confidence = float(entry.confidence)
                confidence_str = f"{confidence:.2f}"
            except (TypeError, ValueError):
                confidence_str = "N/A"
            
            details_text = (
                f"<b>Hora:</b> {time_str}<br>"
                f"<b>Camara:</b> {entry.camera_name} (ID: {entry.camera_id})<br>"
                f"<b>Cara:</b> {entry.face_name}<br>"
                f"<b>Edad:</b> {entry.age if entry.age else 'N/A'}<br>"
                f"<b>Genero:</b> {entry.gender if entry.gender else 'N/A'}<br>"
                f"<b>Confianza:</b> {confidence_str}<br>"
            )
            self.details_label.setText(details_text)
            self.view_screenshot_btn.setEnabled(
                entry.screenshot_path is not None and 
                len(str(entry.screenshot_path)) > 0
            )
            
        except Exception as e:
            logger.error(f"Error al mostrar el elemento del historial: {e}")
            self.details_label.setText("Error al cargar los detalles de la entrada")
            self.view_screenshot_btn.setEnabled(False)

    def view_screenshot(self):
        if self.current_entry is None or not self.current_entry.screenshot_path:
            QMessageBox.information(self, "Sin captura de pantalla", "No hay captura de pantalla disponible para esta entrada")
            return
            
        try:
            screenshot_path = Path(self.current_entry.screenshot_path)
            if not screenshot_path.exists():
                QMessageBox.warning(self, "Archivo Faltante", f"Archivo de captura no encontrado: {screenshot_path}")
                return
                
            image = cv2.imread(str(screenshot_path))
            if image is None:
                raise ValueError("No se pudo leer la captura de pantalla")
                
            pixmap = numpy_to_pixmap(image)
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Captura de Pantalla")
            layout = QVBoxLayout()
            
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(image_label)
            
            close_btn = QPushButton("Cerrar")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            logger.error(f"Error viewing screenshot: {e}")
            QMessageBox.critical(self, "Error", f"No se pudo cargar la captura de pantalla: {str(e)}")