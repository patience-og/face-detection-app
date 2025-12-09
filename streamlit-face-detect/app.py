import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title("Face Detection App (Viola–Jones Algorithm)")
st.markdown("### Instructions:")
st.write("""
**How to use this app:**
1. Upload an image.
2. Choose the rectangle color for detected faces.
3. Adjust *scaleFactor* and *minNeighbors* for better detection.
4. Click on the image to view detected faces.
5. Download the final output image.
""")

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Rectangle color picker
color = st.color_picker("Select Rectangle Color", "#00FF00")
bgr_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

# Sliders for detection parameters
scale_factor = st.slider("scaleFactor", 1.01, 1.5, 1.1)
min_neighbors = st.slider("minNeighbors", 1, 10, 5)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_np, (x, y), (x + w, y + h), bgr_color, 2)

    st.image(img_np, caption="Detected Faces", use_column_width=True)

    # Save output
    result_image = Image.fromarray(img_np)
    output_path = "detected_faces.jpg"
    result_image.save(output_path)

    # Download button
    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Image",
            data=f,
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )
