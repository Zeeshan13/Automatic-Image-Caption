# Automated Image Captioning
This project demonstrates an image captioning system built using a CNN-RNN model architecture, designed to generate descriptive captions for images. The model utilizes TensorFlow and Keras, and the app is implemented in Streamlit to provide an interactive user experience.

## Project Overview
The goal of this project is to create accurate and meaningful captions for images by using a dual-network approach, combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).

## Dataset
- Flickr8k Dataset
- Captions provided as a text file with mappings from image IDs to captions.

## Key Components
1. **Feature Extraction**: VGG16 extracts feature vectors from images.
2. **Caption Preprocessing**: Clean and tokenize captions, adding start and end tokens.
3. **Model Architecture**: Combines image and text processing paths with Dense, Embedding, and LSTM layers.
4. **Training**: Model is trained with a data generator.
5. **Evaluation**: BLEU scores are calculated to evaluate performance.

## Dataset Preprocessing

+ **Image Features:** The dataset used for training is preprocessed by extracting high-level features from images, leveraging a pre-trained CNN model such as InceptionV3.
  
+ **Caption Processing:** Text captions are tokenized and converted to sequences, and start and end tokens are added for consistency in the generated captions.


## Model Architecture

**1. CNN Encoder**
   
+ Uses a pre-trained InceptionV3 model to extract features from images, providing a vectorized representation of visual content.
+ The extracted features are reshaped to fit the requirements of the RNN decoder.

**2. RNN Decoder**
+ The model’s RNN layer, specifically an LSTM, sequentially generates captions based on the CNN-encoded features.
+ The decoder uses word embeddings to convert tokens to dense vectors, and the generated captions are refined by processing these embeddings.

**3. Embedding and Sequence Generation**
+ Word embeddings are used to transform tokens into dense vectors, enabling the model to capture semantic relationships between words.
+ Sequences of words are generated until an end token is reached, producing a coherent caption.

## Training Strategy

+ **Checkpointing:** Model checkpoints are saved periodically to allow resuming from the last saved point in case of interruptions.
+ **Data Augmentation:** Techniques like resizing and normalization are applied to the input images to improve generalization.
+ **Loss and Metrics:** The model is trained with sparse categorical cross-entropy as the loss function, optimizing both accuracy and fluency.

## Model Inference
The trained model generates captions for new images by passing the image through the encoder (CNN) and using the decoder (RNN) to generate text sequentially. The system can process images uploaded through the Streamlit interface, displaying both generated and actual captions when available.

## Key Features

+ **Interactive User Interface:** A Streamlit app for uploading images and generating captions in real-time.
+ **Model Checkpointing:** Saves model checkpoints during training to prevent data loss in case of interruptions.
+ **Generated vs. Actual Captions:** Displays both generated and actual captions (if available) to assess model performance.


## Access the Application
You can access the live application here: [Automatic Image Captioning App](https://automatic-image-caption-5uhujjwfa7z6mqyje99ggb.streamlit.app/)

## Future Work

Potential improvements include:

+ **Exploring Transformer Models:** Testing Transformer-based architectures to further improve caption quality and capture contextual nuances.
+ **Dataset Expansion:** Leveraging larger and more diverse datasets to enhance vocabulary and generalization.
+ **Beam Search for Caption Generation:** Implementing beam search during inference for more accurate caption generation.

## Conclusion

This project showcases the effectiveness of CNN-RNN architectures in generating descriptive captions for images. The integration of pre-trained image processing models and sequential RNN decoders enables a robust framework for generating meaningful captions that reflect image content accurately.
