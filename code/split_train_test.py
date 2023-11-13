import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from scipy.special import softmax


def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets.

    Args:
        df (DataFrame): The DataFrame to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)

def create_hf_datasets(train_df, test_df):
    """
    Converts pandas DataFrames into HuggingFace datasets.

    Args:
        train_df (DataFrame): The training DataFrame.
        test_df (DataFrame): The testing DataFrame.

    Returns:
        DatasetDict: A HuggingFace DatasetDict containing the train and test datasets.
    """
    return datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "test": datasets.Dataset.from_pandas(test_df)
    })

def tokenize_data(dataset, model_checkpoint):
    """
    Tokenizes the dataset using a specified model checkpoint.

    Args:
        dataset (DatasetDict): The HuggingFace DatasetDict to tokenize.
        model_checkpoint (str): The model checkpoint to use for tokenization.

    Returns:
        DatasetDict: The tokenized dataset.
        AutoTokenizer: The tokenizer used.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='longest')

    return dataset.map(preprocess_function, batched=True), tokenizer

def train_model(train_dataset, eval_dataset, model_checkpoint, training_args):
    """
    Trains the model using the specified datasets, model checkpoint, and training arguments.

    Args:
        train_dataset (Dataset): The dataset for training the model.
        eval_dataset (Dataset): The dataset for evaluating the model during training.
        model_checkpoint (str): The pre-trained model checkpoint to start from.
        training_args (TrainingArguments): The configuration for training.

    Returns:
        Trainer: An instance of the Trainer class with the trained model.
    """
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return {'accuracy': accuracy_score(labels, predictions)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )

    data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_checkpoint))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer

def evaluate_model(trainer, dataset):
    """
    Evaluates the model on a given dataset and prints the confusion matrix and accuracy.

    Args:
        trainer (Trainer): The trained model's trainer instance.
        dataset (Dataset): The dataset to evaluate, wrapped in a HuggingFace Dataset object.

    Returns:
        tuple: A tuple containing the confusion matrix and accuracy score.
    """
    predictions_output = trainer.predict(dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    true_labels = dataset['label']
    cm = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc}")
    return cm, acc

def compute_softmax_probabilities(logits):
    """
    Computes softmax probabilities from logits.

    Args:
        logits (np.ndarray): Logits output from the model.

    Returns:
        np.ndarray: Softmax probabilities for each class.
    """
    return softmax(logits, axis=1)

def predict_text(trainer, tokenizer, text):
    """
    Predicts the class probabilities of a given text using the trained model.

    Args:
        trainer (Trainer): The trained model's trainer instance.
        tokenizer (AutoTokenizer): The tokenizer used for the model.
        text (str): The text to classify.

    Returns:
        np.ndarray: The softmax probability distribution over classes for the text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = trainer.model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    probabilities = compute_softmax_probabilities(logits)
    return probabilities


def generate_predictions_dataframe(trainer, tokenized_dataset):
    """
    Generates a DataFrame containing texts, their real labels, predicted labels, and softmax probabilities.

    Args:
        trainer (Trainer): The trained model's trainer instance.
        tokenized_dataset (DatasetDict): The tokenized dataset used for predictions.

    Returns:
        DataFrame: A DataFrame with texts, real labels, predicted labels, and softmax probabilities.
    """
    def get_predictions(dataset, dataset_name):
        texts = [item['text'] for item in dataset]
        true_labels = [item['label'] for item in dataset]
        predictions_output = trainer.predict(dataset)
        logits = predictions_output.predictions
        predictions = np.argmax(logits, axis=-1)
        probabilities = compute_softmax_probabilities(logits)

        return pd.DataFrame({
            'text': texts,
            'test_or_train': [dataset_name] * len(texts),
            'real_label': true_labels,
            'model_label': predictions,
            'probability_class_0': probabilities[:, 0],
            'probability_class_1': probabilities[:, 1],
        })

    train_df = get_predictions(tokenized_dataset["train"], 'train')
    test_df = get_predictions(tokenized_dataset["test"], 'test')

    return pd.concat([train_df, test_df], ignore_index=True)


def gen_model(df, model_checkpoint, training_args):
    """
    Main function to execute the model training and evaluation workflow.

    Args:
        df (DataFrame): The DataFrame containing the dataset.
        model_checkpoint (str): The model checkpoint to use for training and tokenization.
        training_args (TrainingArguments): The training arguments for the model.

    Returns:
        Trainer: The trained model's trainer instance.
        AutoTokenizer: The tokenizer used for the model.
    """
    train_df, test_df = split_data(df)
    dataset_dict = create_hf_datasets(train_df, test_df)
    tokenized_dataset, tokenizer = tokenize_data(dataset_dict, model_checkpoint)
    trainer = train_model(tokenized_dataset["train"], tokenized_dataset["test"], model_checkpoint, training_args)
    
    # Evaluate model on both training and testing data
    evaluate_model(trainer, tokenized_dataset["train"])
    evaluate_model(trainer, tokenized_dataset["test"])
    
    # Generate prediction results dataframe and save to CSV
    results_df = generate_predictions_dataframe(trainer, tokenized_dataset)
    results_df.to_csv("../results/model_predictions.csv", index=False)
    return trainer, tokenizer
