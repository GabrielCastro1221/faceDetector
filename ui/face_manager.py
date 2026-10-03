import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QFileDialog, QMessageBox, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from loguru import logger
import cv2
import numpy as np
from pathlib import Path

from core.utils import numpy_to_pixmap, resize_image

class FaceManagerDialog(QDialog):
    def __init__(self, face_detector, known_faces_dir):
        super().__init__()
        self.face_detector = face_detector
        self.known_faces_dir = known_faces_dir
        self.current_image = None
        
        self.setWindowTitle("Administrador de rostros")
        self.setGeometry(200, 200, 800, 600)
        
        self.init_ui()
        self.load_face_list()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        top_layout = QHBoxLayout()
        
        self.face_list = QListWidget()
        self.face_list.currentItemChanged.connect(self.on_face_selected)
        top_layout.addWidget(self.face_list, 3)
        
        self.face_preview = QLabel()
        self.face_preview.setAlignment(Qt.AlignCenter)
        self.face_preview.setMinimumSize(300, 300)
        top_layout.addWidget(self.face_preview, 2)
        
        layout.addLayout(top_layout)
        
        middle_layout = QHBoxLayout()
        
        name_layout = QVBoxLayout()
        name_layout.addWidget(QLabel("Nombre:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        middle_layout.addLayout(name_layout)
        
        layout.addLayout(middle_layout)
        
        button_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("Agregar Rostro")
        self.add_btn.clicked.connect(self.add_face)
        button_layout.addWidget(self.add_btn)
        
        self.update_btn = QPushButton("Actualizar Rostro")
        self.update_btn.clicked.connect(self.update_face)
        button_layout.addWidget(self.update_btn)
        
        self.delete_btn = QPushButton("Eliminar Rostro")
        self.delete_btn.clicked.connect(self.delete_face)
        button_layout.addWidget(self.delete_btn)
        
        self.import_btn = QPushButton("Importar Imagen")
        self.import_btn.clicked.connect(self.import_image)
        button_layout.addWidget(self.import_btn)
        
        self.close_btn = QPushButton("Cerrar")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def load_face_list(self):
        self.face_list.clear()
        known_faces_dir = Path(self.known_faces_dir)
        
        if not known_faces_dir.exists():
            logger.warning(f"Directorio de caras conocidas {known_faces_dir} no existe")
            return
            
        for face_file in known_faces_dir.glob('*.*'):
            if face_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.face_list.addItem(face_file.stem)
                
    def on_face_selected(self, current, previous):
        if current is None:
            self.face_preview.clear()
            self.name_input.clear()
            return
            
        face_name = current.text()
        self.name_input.setText(face_name)
        
        face_path = Path(self.known_faces_dir) / f"{face_name}{self.get_face_extension(face_name)}"
        if not face_path.exists():
            QMessageBox.warning(self, "Error", f"Archivo de imagen no encontrado: {face_path}")
            return
            
        try:
            image = cv2.imread(str(face_path))
            if image is None:
                raise ValueError("No se pudo leer la imagen")
                
            self.current_image = image
            pixmap = numpy_to_pixmap(image)
            self.face_preview.setPixmap(pixmap.scaled(
                self.face_preview.width(), self.face_preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar imagen: {str(e)}")
            logger.error(f"Error al cargar la imagen facial: {e}")
            
    def get_face_extension(self, face_name: str) -> str:
        known_faces_dir = Path(self.known_faces_dir)
        for ext in ['.jpg', '.jpeg', '.png']:
            if (known_faces_dir / f"{face_name}{ext}").exists():
                return ext
        return ''
        
    def add_face(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Por favor ingrese un nombre para la cara")
            return
            
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "Por favor importe o seleccione una imagen primero")
            return
            
        existing_files = list(Path(self.known_faces_dir).glob(f"{name}.*"))
        if existing_files:
            QMessageBox.warning(self, "Error", f"Una cara con el nombre '{name}' ya existe")
            return
            
        success = self.face_detector.add_known_face(
            self.current_image, name, self.known_faces_dir)
            
        if success:
            QMessageBox.information(self, f"Rostro '{name}' añadido exitosamente")
            self.load_face_list()
        else:
            QMessageBox.warning(self, "Error", "No se pudo agregar la cara. Asegúrese de que la imagen contenga un rostro claro y visible.")
            
    def update_face(self):
        current_item = self.face_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "Error", "Por favor seleccione una cara para actualizar")
            return
            
        old_name = current_item.text()
        new_name = self.name_input.text().strip()
        
        if not new_name:
            QMessageBox.warning(self, "Error", "Por favor ingrese un nombre para la cara")
            return
            
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "Por favor importe o seleccione una imagen primero")
            return
            
        if old_name != new_name:
            old_path = Path(self.known_faces_dir) / f"{old_name}{self.get_face_extension(old_name)}"
            new_path = Path(self.known_faces_dir) / f"{new_name}{old_path.suffix}"
            
            if new_path.exists():
                QMessageBox.warning(self, "Error", f"Una cara con el nombre '{new_name}' ya existe")
                return
                
            try:
                old_path.rename(new_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cambiar el nombre de la cara: {str(e)}")
                return
                
        try:
            current_path = Path(self.known_faces_dir) / f"{new_name}{self.get_face_extension(new_name)}"
            cv2.imwrite(str(current_path), self.current_image)
            
            self.face_detector.load_known_faces(self.known_faces_dir)
            
            QMessageBox.information(self, f"Rostro '{new_name}' actualizado exitosamente")
            self.load_face_list()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo actualizar la cara: {str(e)}")

    def delete_face(self):
        current_item = self.face_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "Error", "Por favor seleccione una cara para eliminar")
            return
            
        name = current_item.text()
        
        reply = QMessageBox.question(
            self, "Confirmar eliminación",
            f"¿Está seguro de que desea eliminar el rostro '{name}'?",
            QMessageBox.Yes | QMessageBox.No)
            
        if reply == QMessageBox.No:
            return
            
        face_path = Path(self.known_faces_dir) / f"{name}{self.get_face_extension(name)}"
        try:
            face_path.unlink()

            self.face_detector.load_known_faces(self.known_faces_dir)
            self.load_face_list()
            
            QMessageBox.information(self, "Success", f"Rostro '{name}' eliminado exitosamente")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo eliminar la cara: {str(e)}")
            
    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)")
            
        if not file_path:
            return
            
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("No se pudo leer la imagen")
                
            self.current_image = image
            pixmap = numpy_to_pixmap(image)
            self.face_preview.setPixmap(pixmap.scaled(
                self.face_preview.width(), self.face_preview.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
            suggested_name = Path(file_path).stem
            self.name_input.setText(suggested_name)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar la imagen: {str(e)}")
            logger.error(f"Error importing image: {e}")