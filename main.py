import streamlit as st
import os, cv2
from datetime import datetime
from baumer_camera import CameraManager
from web_cam import WebcamManager
from yolo_model import BoltDetector
from excel_logger import append_result
# Excel helper with image embedding

# --- Setup folders ---
CAPTURE_DIR = "captured_images"
OUTPUT_DIR = "outputs"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = BoltDetector(model_path="gray_scale.pt")

st.title("Vehicle Bolt Detection")

if "camera_mgr" not in st.session_state:
    st.session_state.camera_mgr = WebcamManager()
    try:
        st.session_state.camera_mgr.start()
        st.success("Camera initialized successfully.")
    except Exception as e:
        st.error(f"Camera initialization failed: {e}")
        st.session_state.camera_mgr = None

camera_mgr = st.session_state.camera_mgr

vin_number = st.text_input("VIN Number")
vehicle_model = st.text_input("Model")

col1, col2 = st.columns(2)

with col1:
    if st.button("CAPTURE"):
        if camera_mgr and camera_mgr.is_initialized():
            if not vin_number:
                st.warning("Please enter a VIN number before capturing.")
            else:
                try:
                    frame = camera_mgr.capture_frame()
                    bgr = frame
                    filename = f"{vin_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    path = os.path.join(CAPTURE_DIR, filename)
                    cv2.imwrite(path, bgr)
                    st.session_state.last_captured = path
                    st.image(bgr, channels="BGR", caption="Captured Image")
                except Exception as e:
                    st.error(f"Capture failed: {e}")
        else:
            st.error("Camera not ready.")

with col2:
    if st.button("DETECT"):
        try:
            files = os.listdir(CAPTURE_DIR)
            if not files:
                st.warning("No captured image found.")
            else:
                # last_file = sorted(files)[-1]
                img_path = st.session_state.last_captured
                img = cv2.imread(img_path)
                # img_path = r"C:\Users\LENOVA\Downloads\20250926_123744_IMG-20250920-WA0056.jpg"
                # img = cv2.imread(img_path)
                # img_path = os.path.join(CAPTURE_DIR, last_file)
                # img = cv2.imread(img_path)

                # --- Run detection ---
                detected_img, class_counts = detector.detect_and_draw(img)
                detected_path = detector.save_detected_image(detected_img,
                                                             output_dir=OUTPUT_DIR,
                                                             prefix=f"detected_{vin_number}")
                st.image(detected_img, channels="BGR", caption="Detected Image")

                # --- Determine Poke Yoke Status ---
                poke_yoke_status = "OK" if class_counts.get("fixed_bolt", 0) == 4 else "NOT OK"

                # --- Save to Excel ---
                append_result(
                    vin_number=vin_number,
                    model_name=vehicle_model,
                    class_counts=class_counts,
                    image_path=detected_path
                )
                st.success(f"Detection complete. Poke Yoke Status: {poke_yoke_status}")
        except Exception as e:
            st.error(f"Detection failed: {e}")
