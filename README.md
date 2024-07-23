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

Developed sentiment analysis on social media posts using NLTK for preprocessing and LSTM networks for classification. The model classifies data into positive or negative sentiment, achieving 99% accuracy on the training data and 89% accuracy on the test data.

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

Contributions are welcome! If you have suggestions, improvements, or additional content to contribute, feel free to open issues, submit pull requests, or provide feedback. 

[![GitHub watchers](https://img.shields.io/github/watchers/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Watch)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/watchers/?WT.mc_id=academic-105485-koreyst)
[![GitHub forks](https://img.shields.io/github/forks/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Fork)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/network/?WT.mc_id=academic-105485-koreyst)
[![GitHub stars](https://img.shields.io/github/stars/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Star)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/stargazers/?WT.mc_id=academic-105485-koreyst)

## Author

This repository is maintained by Elsayed Elmandoh, an AI Engineer. You can connect with Elsayed on [LinkedIn and Twitter/X](https://linktr.ee/elsayedelmandoh) for updates and discussions related to Machine learning, deep learning and NLP.

Happy coding!

