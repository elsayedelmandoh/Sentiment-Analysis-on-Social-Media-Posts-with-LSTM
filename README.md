# Sentiment Analysis with LSTM

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [Author](#author)


## Overview

This project implements sentiment analysis on social media posts using Long Short-Term Memory (LSTM) networks. The primary goal is to build a model capable of classifying text data into sentiment classes, such as positive or negative.

## Dataset

We utilized the [Amazon Reviews dataset](https://www.kaggle.com/bittlingmayer/amazonreviews), which consists of 3.6 million reviews with binary sentiment classes (positive or negative). The dataset was downloaded using the Kaggle API.

## Project Structure

The project follows a structured approach, including the following key steps:

1. **Data Acquisition:** Downloading and preparing the dataset for training and testing.

2. **Data Exploration:** Understanding the dataset's characteristics, checking for missing values, duplicated rows, and analyzing the distribution of text lengths.

3. **Data Preparation:** Applying text preprocessing techniques, splitting data into training and testing sets, and tokenizing/padding text sequences.

4. **LSTM Model:** Building an LSTM model using TensorFlow/Keras with layers like Embedding, LSTM, Dropout, and Dense.

5. **Training the Model:** Training the model on the prepared data and monitoring performance metrics.

6. **Metrics Evaluation:** Calculating accuracy, generating a classification report, and plotting a confusion matrix to evaluate model performance.

## Requirements

- Python 3.x
- TensorFlow
- Pandas
- Numpy
- NLTK
- Matplotlib
- Seaborn

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/elsayedelmandoh/Sentiment-Analysis-on-Social-Media-Posts-with-LSTM.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook or Python script to execute the sentiment analysis project.

## Contributing
  Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback.

## Author
  Elsayed Elmandoh : [Linkedin](https://www.linkedin.com/in/elsayed-elmandoh-77544428a/).


