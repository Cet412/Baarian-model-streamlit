import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import platform

# Load model
model = YOLO(r"Baarian_Model.pt")

st.title("📷 Deteksi Bahasa Isyarat - BISINDO")
st.markdown("Upload gambar atau gunakan kamera (jika di lokal).")

# Deteksi apakah sedang di local atau cloud
is_local = platform.system() == "Windows" or platform.system() == "Linux"

if is_local:
    img = st.camera_input("Ambil gambar dari kamera")
else:
    img = st.file_uploader("Upload gambar dari kamera", type=["jpg", "jpeg", "png"])

if img is not None:
    image = Image.open(img)
    st.image(image, caption="Gambar", use_container_width=True)

    img_np = np.array(image)
    img_bgr = img_np[:, :, ::-1]
    results = model.predict(img_bgr, conf=0.7, verbose=False)
    result = results[0]

    annotated = result.plot()
    st.image(annotated, caption="Hasil Deteksi", use_container_width=True)

    if len(result.boxes) > 0:
        detected_class = result.names[int(result.boxes[0].cls.item())]
        st.success(f"Huruf terdeteksi: **{detected_class}**")
    else:
        st.warning("❌ Tidak ada huruf terdeteksi.")