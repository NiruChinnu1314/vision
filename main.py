import base64
import io

import requests
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
EXTERNAL_LOGGING_API = "https://tpjtb58k-9823.use.devtunnels.ms/api/live/updates"


detector = BoltDetector(model_path="gray_scale.pt")

st.title("Vehicle Bolt Detection")

vin_number = st.text_input("VIN Number")
vehicle_model = st.text_input("Model")

# --- Camera Input from Browser ---
st.subheader("Capture Image from Webcam")
camera_image = st.camera_input("Take a picture")

def post_inspection_data(api_url, vin, model, class_counts, poke_status, captured_path, detected_path):
    """Sends inspection results and base64-encoded images as JSON to an external REST API."""

    def encode_image_to_base64(image_path, max_size=(640, 640), quality=50):
        img = Image.open(image_path)
        img.thumbnail(max_size)  # Resize
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)  # Compress
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 1. Encode images
    captured_b64 = encode_image_to_base64(captured_path)
    detected_b64 = encode_image_to_base64(detected_path)

    # 2. Prepare payload
    payload = {
        "vin_number": vin,
        "model_name": model,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "inspection_status": poke_status,
        "bolt_counts": {
            "loose_bolts": class_counts.get("loose_bolt", 0),
            "fixed_bolts": class_counts.get("fixed_bolt", 0),
            "no_bolts": class_counts.get("no_bolt", 0)
        },
        "captured_image_base64": captured_b64,
        "detected_image_base64": detected_b64
    }

    st.subheader("Outgoing API JSON Payload")
    st.json(payload)

    try:
        # 3. Send JSON payload
        response = requests.post(api_url, json=payload, timeout=15)
        response.raise_for_status()
        return True, f"✅ API logged successfully (Status {response.status_code})"

    except requests.exceptions.RequestException as api_e:
        return False, f"❌ Failed to log data: {api_e}"


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

            # --- Run detection ---
            detected_img, class_counts = detector.detect_and_draw(img)
            detected_path = detector.save_detected_image(
                detected_img,
                output_dir=OUTPUT_DIR,
                prefix=f"detected_{vin_number}"
            )
            st.image(detected_img, channels="BGR", caption="Detected Image")

            # --- Determine Poke Yoke Status ---
            poke_yoke_status = "OK" if class_counts.get("fixed_bolt", 0) == 4 else "NOT OK"

            if EXTERNAL_LOGGING_API and os.path.exists(img_path) and os.path.exists(detected_path):
                success, message = post_inspection_data(
                    EXTERNAL_LOGGING_API,
                    vin_number,
                    vehicle_model,
                    class_counts,
                    poke_yoke_status,
                    img_path,
                    detected_path
                )
                if success:
                    st.info(message)
                else:
                    st.error(message)
            else:
                st.warning("API endpoint or required image paths are missing. Skipping POST request.")

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

        st.error(f"Detection failed: {e}")



