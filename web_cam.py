# webcam_camera.py
import cv2
import time
from threading import Thread, Lock


class WebcamManager:
    def __init__(self, cam_index=0):
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.lock = Lock()
        self.cam_index = cam_index
        self.initialized = False

    def is_initialized(self):
        return self.initialized and self.running

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {self.cam_index}")

        self.running = True
        self.initialized = True
        print("Webcam started.")
        Thread(target=self._grab_frames, daemon=True).start()

    def _grab_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                print("Frame grab failed.")
            time.sleep(0.01)

    def capture_frame(self):
        if not self.initialized:
            raise RuntimeError("Webcam not initialized.")
        start = time.time()
        while time.time() - start < 5:
            with self.lock:
                if self.latest_frame is not None:
                    return self.latest_frame.copy()
            time.sleep(0.05)
        raise RuntimeError("No frame captured within timeout.")

    def stop(self):
        self.running = False
        self.initialized = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.cap = None
        print("Webcam stopped.")