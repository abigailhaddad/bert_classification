# Text Classification Sample Project

## Overview

This project demonstrates a simple binary text classification task using a sample dataset. It includes scripts to generate sample data (very short recipes and job descriptions) and to train a binary classification model on this data.

## Scripts

1. **Data Generation (`gen_test_data.py`)**: This script contains functions to generate sample data, creating two types of short texts: recipes and job descriptions. These texts are then combined into a pandas DataFrame with corresponding labels.

2. **Model Training and Prediction (`split_train_test_sample.py`)**: This script trains a binary classification model using the Hugging Face Transformers library. It handles splitting the data, tokenizing it, training the model, evaluating its performance, and predicting classes for new texts.

## How to Use

1. **Data Generation**:
    - Run `gen_test_data.py` to create a DataFrame of sample texts and their labels, which is shuffled and ready for model training.

2. **Model Training**:
    - Configure the model checkpoint and training arguments in `split_train_test_sample.py`.
    - Run the script to train the model. It splits the data, tokenizes the texts, and trains the classification model.
    - It also evaluates the model on the test set and outputs a confusion matrix and accuracy score.
    - Predictions and softmax probabilities are saved to a CSV file.

3. **Prediction**:
    - Use the `predict_text` function to predict the class of new text samples with the trained model.

## Dependencies

Install the required packages from the `requirements.txt` file:

```sh
pip install -r requirements.txt
