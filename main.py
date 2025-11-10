import streamlit as st
import os, cv2
from datetime import datetime
from yolo_model import BoltDetector
from excel_logger import append_result
from PIL import Image
import numpy as np

# --- Setup folders ---
CAPTURE_DIR = "captured_images"
OUTPUT_DIR = "outputs"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = BoltDetector(model_path="gray_scale.pt")

st.title("Vehicle Bolt Detection")

vin_number = st.text_input("VIN Number")
vehicle_model = st.text_input("Model")

# --- Camera Input from Browser ---
st.subheader("Capture Image from Webcam")
camera_image = st.camera_input("Take a picture")

# --- Capture Button (Save the photo to disk) ---
if camera_image is not None:
    if not vin_number:
        st.warning("Please enter a VIN number before capturing.")
    else:
        # Convert image to OpenCV format
        img = Image.open(camera_image)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        filename = f"{vin_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(CAPTURE_DIR, filename)
        cv2.imwrite(path, img)
        st.session_state.last_captured = path
        st.image(img, channels="BGR", caption="Captured Image Saved ✅")

# --- Detection Button ---
if st.button("DETECT"):
    try:
        if "last_captured" not in st.session_state:
            st.warning("No image captured yet.")
        else:
            img_path = st.session_state.last_captured
            img = cv2.imread(img_path)

            # Run detection
            detected_img, class_counts = detector.detect_and_draw(img)
            detected_path = detector.save_detected_image(
                detected_img, output_dir=OUTPUT_DIR, prefix=f"detected_{vin_number}"
            )

            st.image(detected_img, channels="BGR", caption="Detected Image")

            # Determine Status
            poke_yoke_status = "OK" if class_counts.get("fixed_bolt", 0) == 4 else "NOT OK"

            # Log to Excel
            append_result(
                vin_number=vin_number,
                model_name=vehicle_model,
                class_counts=class_counts,
                image_path=detected_path,
            )

            st.success(f"Detection complete. Poke Yoke Status: {poke_yoke_status}")

    except Exception as e:
        st.error(f"Detection failed: {e}")


