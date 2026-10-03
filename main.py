import sys
import yaml
from pathlib import Path
from loguru import logger
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

from ui.main_window import MainWindow

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        Path(config['app']['screenshot_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['app']['known_faces_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['app']['log_dir']).mkdir(parents=True, exist_ok=True)
        
        return config
    except Exception as e:
        logger.error(f"No se pudo cargar la configuración: {e}")
        raise

def setup_logging(log_dir: str):
    logger.add(
        f"{log_dir}/app.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO"
    )
    logger.add(
        f"{log_dir}/error.log",
        rotation="10 MB",
        retention="30 days",
        level="ERROR"
    )


def show_splash_screen(config: dict) -> QSplashScreen:
    try:
        logo_path = config.get('app', {}).get('logo', 'assets/logo.png')
        
        if not Path(logo_path).exists():
            raise FileNotFoundError(f"Archivo de logotipo no encontrado: {logo_path}")
        
        splash_pix = QPixmap(logo_path)
        if splash_pix.isNull():
            raise ValueError(f"Imagen de logotipo inválida: {logo_path}")
            
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setMask(splash_pix.mask())

        splash.showMessage(
            "Inicializando Face Tracker...",
            Qt.AlignBottom | Qt.AlignCenter,
            Qt.white
        )
        QApplication.processEvents()
        
        return splash
        
    except Exception as e:
        print(f"Error al cargar la pantalla de presentación: {e}")
        return QSplashScreen(QPixmap(800, 400))


def main():
    try:
        config = load_config('config/config.yaml')
        
        setup_logging(config['app']['log_dir'])
        
        app = QApplication(sys.argv)

        splash = show_splash_screen(config)
        splash.show()

        app.processEvents()

        window = MainWindow(config)
        
        def show_ai_loading():
            splash.showMessage(
                "Cargando modelos de IA...",
                Qt.AlignBottom | Qt.AlignCenter,
                Qt.white
            )
            QApplication.processEvents()
            
            QTimer.singleShot(1500, lambda: splash.finish(window))
            
        QTimer.singleShot(2000, show_ai_loading)

        QTimer.singleShot(3500, lambda: window.show())


        logger.info("La aplicación se inició con éxito")
        def on_close():
            window.camera_manager.stop_all_cameras()
            window.alert_system.shutdown()
            app.quit()
            
        app.aboutToQuit.connect(on_close)

        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"La aplicación falló al iniciar: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()