from gen_test_data import create_dataframe
from split_train_test import gen_model, predict_text
from transformers import TrainingArguments

if __name__ == "__main__":
    # Generate a dataframe with test data
    df = create_dataframe()

    # Define model checkpoint and training arguments
    model_checkpoint = "microsoft/deberta-v3-base"
    training_args = TrainingArguments(
        output_dir='../model/deberta-v3-base-classification',
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,  # Frequency of logging
    )

    # Train and evaluate the model
    trainer, tokenizer = gen_model(df, model_checkpoint, training_args)

    # Predict the class of a specific text
    text_to_predict = "Your example text here"
    predicted_class = predict_text(trainer, tokenizer, text_to_predict)
    print(f"Predicted class: {predicted_class}")
