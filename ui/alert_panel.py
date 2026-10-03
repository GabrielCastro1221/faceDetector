from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt
from loguru import logger
import time

class AlertPanel(QDialog):
    def __init__(self, alert_system):
        super().__init__()
        self.alert_system = alert_system
        self.setWindowTitle("Panel de alerta")
        self.setGeometry(300, 300, 600, 400)
        
        self.init_ui()
        self.load_alerts()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.alert_list = QListWidget()
        layout.addWidget(self.alert_list)
        
        controls_layout = QHBoxLayout()
        
        self.enable_alerts_check = QCheckBox("Habilitar alertas")
        self.enable_alerts_check.setChecked(self.alert_system.alert_enabled)
        self.enable_alerts_check.stateChanged.connect(self.toggle_alerts)
        controls_layout.addWidget(self.enable_alerts_check)
        
        self.enable_screenshots_check = QCheckBox("Habilitar capturas de pantalla")
        self.enable_screenshots_check.setChecked(self.alert_system.screenshot_enabled)
        self.enable_screenshots_check.stateChanged.connect(self.toggle_screenshots)
        controls_layout.addWidget(self.enable_screenshots_check)
        
        self.clear_btn = QPushButton("Limpiar alertas")
        self.clear_btn.clicked.connect(self.clear_alerts)
        controls_layout.addWidget(self.clear_btn)
        
        self.close_btn = QPushButton("Cerrar")
        self.close_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.close_btn)
        
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
        
    def load_alerts(self):
        self.alert_list.clear()
        alerts = self.alert_system.get_recent_alerts(50)
        
        for alert in alerts:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))
            item_text = f"{time_str} - {alert.face_name} on {alert.camera_name} (Confidence: {alert.confidence:.2f})"
            self.alert_list.addItem(item_text)
            
    def toggle_alerts(self, state):
        self.alert_system.enable_alerts(state == Qt.Checked)
        
    def toggle_screenshots(self, state):
        self.alert_system.enable_screenshots(state == Qt.Checked)
        
    def clear_alerts(self):
        reply = QMessageBox.question(
            self, "Confirm Clear",
            "¿Está seguro de que desea limpiar todas las alertas?",
            QMessageBox.Yes | QMessageBox.No)
            
        if reply == QMessageBox.Yes:
            self.alert_system.clear_alerts()
            self.alert_list.clear()