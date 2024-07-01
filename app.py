import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="MobileNetV2 Image Classifier", page_icon="🖼️")

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

def preprocess_image(img):
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict(img):
    x = preprocess_image(img)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

# Load the model
model = load_model()

# Set up the Streamlit app
st.title('📱 MobileNetV2 Image Classifier')
st.write("Upload a photo and our AI will classify what's in it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    with st.spinner('Analyzing the image...'):
        predictions = predict(image)
    
    # Display results
    st.subheader("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i+1}. {label.title()} ({score:.2%})")
        st.progress(float(score))  # Convert score to Python float

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This app uses MobileNetV2, a lightweight deep learning model, "
    "to classify images. It's designed to work well with images taken "
    "from mobile phones. Upload any image to see what the AI thinks it contains!"
)

# Add a footer
st.sidebar.title("Made with")
st.sidebar.write(
    "[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)"
    + " "
    + "[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)"
)
