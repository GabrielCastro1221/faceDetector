import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import time
import threading
import queue
import yaml
from pathlib import Path

@dataclass
class CameraConfig:
    id: int
    name: str
    source: str
    enabled: bool
    width: int
    height: int
    fps: int
    rotate: int

class CameraManager:
    def __init__(self, config_path: str):
        self.cameras: Dict[int, CameraConfig] = {}
        self.capture_threads: Dict[int, threading.Thread] = {}
        self.stop_event = threading.Event()
        self.frame_queues: Dict[int, queue.Queue] = {}
        self.load_config(config_path)

    def _cleanup_camera_thread(self, cam_id: int):
        if cam_id in self.capture_threads:
            self.stop_event.set()
            
            thread = self.capture_threads.pop(cam_id)
            thread.join(timeout=2.0)
            
            if thread.is_alive():
                logger.warning(f"Camera ID {cam_id} thread did not stop gracefully")
            
            if cam_id in self.frame_queues:
                try:
                    while True:
                        self.frame_queues[cam_id].get_nowait()
                except queue.Empty:
                    pass
                del self.frame_queues[cam_id]
                
            logger.debug(f"Cleaned up resources for camera ID {cam_id}")

        
    def load_config(self, config_path: str) -> None:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            self.cameras.clear()
            for cam_config in config.get('cameras', []):
                cam_id = cam_config['id']
                self.cameras[cam_id] = CameraConfig(
                    id=cam_id,
                    name=cam_config.get('name', f'Camera {cam_id}'),
                    source=cam_config['source'],
                    enabled=cam_config.get('enabled', True),
                    width=cam_config['resolution']['width'],
                    height=cam_config['resolution']['height'],
                    fps=cam_config.get('fps', 30),
                    rotate=cam_config.get('rotate', 0)
                )
                
            logger.info(f"Loaded {len(self.cameras)} camera configurations")
            
        except Exception as e:
            logger.error(f"Error loading camera config: {e}")
            raise

    def start_all_cameras(self) -> None:
        self.stop_event.clear()
        for cam_id, cam_config in self.cameras.items():
            if cam_config.enabled:
                self.start_camera(cam_id)
                
    def stop_all_cameras(self) -> None:
        self.stop_event.set()
        for thread in self.capture_threads.values():
            thread.join(timeout=2)
        self.capture_threads.clear()
        self.frame_queues.clear()
        logger.info("All camera threads stopped")

    def start_camera(self, cam_id: int) -> bool:
            if cam_id not in self.cameras:
                logger.error(f"Camera ID {cam_id} not found in configuration")
                return False
                
            if not self.cameras[cam_id].enabled:
                logger.warning(f"Camera ID {cam_id} is disabled in configuration")
                return False
                
            if cam_id in self.capture_threads:
                self._cleanup_camera_thread(cam_id)
                
            self.frame_queues[cam_id] = queue.Queue(maxsize=1)
            self.stop_event.clear() 
            
            self.capture_threads[cam_id] = threading.Thread(
                target=self._capture_frames,
                args=(cam_id,),
                daemon=True,
                name=f"CameraThread-{cam_id}"
            )
            self.capture_threads[cam_id].start()
            logger.info(f"Started camera ID {cam_id}")
            return True
    
    def stop_camera(self, cam_id: int) -> bool:
        if cam_id in self.capture_threads:
            self._cleanup_camera_thread(cam_id)
            logger.info(f"Stopped camera ID {cam_id}")
            return True
        return False

    def _capture_frames(self, cam_id: int) -> None:
        cam_config = self.cameras[cam_id]
        cap = None
        
        try:
            source = int(cam_config.source) if str(cam_config.source).isdigit() else cam_config.source
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open camera ID {cam_id} with source {cam_config.source}")
                return
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config.height)
            cap.set(cv2.CAP_PROP_FPS, cam_config.fps)
            
            logger.info(f"Camera ID {cam_id} opened successfully")
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Camera ID {cam_id} read failed")
                    time.sleep(1)
                    continue
                    
                if cam_config.rotate == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif cam_config.rotate == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif cam_config.rotate == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                if self.frame_queues[cam_id].full():
                    try:
                        self.frame_queues[cam_id].get_nowait()
                    except queue.Empty:
                        pass
                        
                self.frame_queues[cam_id].put(frame)
                
        except Exception as e:
            logger.error(f"Error in camera ID {cam_id} capture thread: {e}")
        finally:
            if cap is not None:
                cap.release()
            logger.info(f"Camera ID {cam_id} capture thread exiting")


    def get_frame(self, cam_id: int) -> Optional[np.ndarray]:
        if cam_id not in self.frame_queues:
            return None
            
        try:
            return self.frame_queues[cam_id].get_nowait()
        except queue.Empty:
            return None

    def get_all_frames(self) -> Dict[int, np.ndarray]:
        frames = {}
        for cam_id in self.frame_queues:
            frame = self.get_frame(cam_id)
            if frame is not None:
                frames[cam_id] = frame
        return frames

    def get_camera_status(self, cam_id: int) -> Dict:
        if cam_id not in self.cameras:
            return {}
            
        status = {
            'id': cam_id,
            'name': self.cameras[cam_id].name,
            'running': cam_id in self.capture_threads,
            'frame_queue_size': self.frame_queues.get(cam_id, queue.Queue()).qsize(),
            'enabled': self.cameras[cam_id].enabled
        }
        return status

    def get_all_camera_status(self) -> List[Dict]:
        return [self.get_camera_status(cam_id) for cam_id in self.cameras]