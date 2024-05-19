import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("LSTM_sentiment_analysis.h5")

# Function to preprocess new text
def preprocess_text(text):
    # Add your preprocessing steps here
    return text

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize and pad the processed text
    max_len = 100  # Same as maxlen used during training
    tokenizer = Tokenizer(num_words=10000)  # Same as tokenizer used during training
    tokenizer.fit_on_texts([processed_text])
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    # Predict sentiment
    prediction = model.predict(padded_sequences)
    
    return prediction[0][0]

# Streamlit UI
st.title("Sentiment Analysis")

# Input text area for user input
text_input = st.text_area("Enter text for sentiment analysis:", "")

# Button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    # Perform sentiment analysis
    sentiment_score = predict_sentiment(text_input)
    
    # Display sentiment score
    # st.write("Sentiment Score:", sentiment_score)
    
    # Categorize sentiment
    if sentiment_score < 0.5:
        st.write("Sentiment: Negative :(")
    else:
        st.write("Sentiment: Positive :)")