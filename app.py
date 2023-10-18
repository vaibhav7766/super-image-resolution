from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageEnhance, ImageFilter
import base64

# Load your Keras model
model = load_model("model1.h5")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    backdrop-filter: blur(1000px);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: white; font-size: 58px; font-family: \'Times New Roman\', Times, serif;'>Enhancing Images</p>", unsafe_allow_html=True)

st.markdown('<p style="font-family: \'Times New Roman\', Times, serif; color: white; font-weight: bold; font-size: 24px; text-align: center;">This app enhances the quality of the input image.</p>', unsafe_allow_html=True)

# st.title("Enhancing Image")

set_background("Computer+Chip+Of+Central+Processing+Unit+Network+Technology+4k+590x.jpg")

# st.info("This app enhances the quality of the input image.")

input_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "image/*"])

if input_image is not None:
    # Open the uploaded image using PIL
    input_pil_image = Image.open(input_image)

    # Resize the image to the desired dimensions
    desired_size = (1920, 1080)
    input_pil_image = input_pil_image.resize(desired_size)

    # Convert the PIL image to a NumPy array
    input_image_array = img_to_array(input_pil_image) / 255.0  # Normalize to [0, 1]
    input_image_array = np.expand_dims(input_image_array, axis=0)  # Add batch dimension

    # Predict the super-resolved image
    super_res_image = model.predict(input_image_array)

    # Convert the NumPy array back to a PIL image
    super_res_image = Image.fromarray(np.uint8(super_res_image[0] * 255.0))

    # Apply image enhancement and sharpening filters
    enhanced_image = ImageEnhance.Contrast(input_pil_image).enhance(2.0)  # Increase contrast
    sharpened_image = input_pil_image.filter(ImageFilter.SHARPEN)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.text("Original Image: ")
        st.image(input_pil_image, use_column_width=True)        
    with col2:
        st.text("Enhanced Image: ")
        st.image(super_res_image, use_column_width=True)
    with col3:
        st.text("Enhancement and Sharpening: ")
        st.image(sharpened_image, use_column_width=True)