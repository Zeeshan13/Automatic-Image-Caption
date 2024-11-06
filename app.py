import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load necessary resources
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    with open("features.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_captioning_model():
    return load_model("RNNmodel.keras")

@st.cache_resource
def load_mapping():
    with open("captions_mapping.pkl", "rb") as f:
        return pickle.load(f)

# Load resources
tokenizer = load_tokenizer()
features = load_features()
model = load_captioning_model()
mapping = load_mapping()
max_length = 34  # Assuming max length from the model training

# Caption generation function
def generate_caption(model, tokenizer, photo, max_length):
    input_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        input_text += " " + word
        if word == "endseq":
            break
    return input_text.split()[1:-1]

# Streamlit App
st.title("Automatic Image Caption")
st.write("Upload an image to see the actual and generated captions")

# Section to show model architectures and BLEU score plots
#st.sidebar.title("Model Architectures and Evaluation")
#st.sidebar.image("Basic-architecture-of-VGG16-model.png", caption="VGG16 Model Architecture", use_column_width=True)
#st.sidebar.image("LSTM Model.png", caption="CNN-RNN Dual Model Architecture", use_column_width=True)
#st.sidebar.image("bleu_scores.png", caption="BLEU Score Plot", use_column_width=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    image = image.resize((299, 299))  # Assuming InceptionV3 or similar feature extraction input
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Extract features from the uploaded image without file extension
    image_key = uploaded_file.name.rsplit('.', 1)[0]  # Remove extension from the file name
    photo_features = features.get(image_key, None)
    
    if photo_features is None:
        st.error("Features for this image are not available in 'features.pkl'.")
    else:
        # Generate caption
        generated_caption = generate_caption(model, tokenizer, photo_features, max_length)
        st.write("### Generated Caption:")
        st.write(" ".join(generated_caption))
        
        # Display actual captions if available in mapping
        actual_captions = mapping.get(image_key, [])
        if actual_captions:
            st.write("### Actual Captions:")
            for caption in actual_captions:
                st.write("- " + caption)
        else:
            st.warning("No actual captions available for this image.")
