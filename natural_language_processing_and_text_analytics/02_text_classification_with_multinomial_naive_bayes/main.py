#!/usr/bin/env python3
"""
MSCI 641 Assignment 2: Text Classification with Multinomial Naive Bayes
-----------------------------------------
Author: Li Xiaohui
Student ID: 21221482
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load tokenized data from a CSV file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [line.strip().split(',') if line.strip() else [] for line in f]
    return data

def load_labels(file_path):
    """Load labels from a CSV file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def train_model(train_data, train_labels, use_bigrams=False, use_unigrams=True):
    """Train a Multinomial Naive Bayes classifier
    
    Args:
        train_data: List of texts to train on
        train_labels: List of corresponding labels
        use_bigrams: Whether to include bigram features
        use_unigrams: Whether to include unigram features
        
    Returns:
        A trained model that can be used for prediction
    
    Implementation steps:
    1. Configure the n-gram range based on the feature requirements
       - If using only unigrams, set range to (1,1)
       - If using only bigrams, set range to (2,2)
       - If using both, set range to (1,2)
       
    2. Create a pipeline with two steps:
       - A text vectorizer that converts text to token counts with the configured n-gram range
       - A Multinomial Naive Bayes classifier
       
    3. Train the pipeline on the training data and labels
    
    4. Return the trained pipeline
    """
    # Your implementation here

    if use_bigrams and use_unigrams:
        ngram_range = (1, 2)
    elif use_unigrams:
        ngram_range = (1, 1)
    elif use_bigrams:
        ngram_range = (2, 2)
    else:
        raise ValueError("At least one of use_bigrams or use_unigrams")

    model = Pipeline([
        ('vect', CountVectorizer(ngram_range=ngram_range)),
        ("Classifier", MultinomialNB())
    ])

    model.fit(train_data, train_labels)

    return model

def evaluate_model(model, test_data, test_labels):
    """Evaluate a model on test data
    
    Args:
        model: A trained classifier model
        test_data: List of texts to evaluate on
        test_labels: List of true labels for the test data
        
    Returns:
        float: The accuracy of the model on the test data
        
    Implementation steps:
    1. Use the model to generate predictions for all samples in the test data
    
    2. Calculate the accuracy by:
       - Comparing each predicted label with the corresponding true label
       - Counting how many predictions match their true labels
       - Dividing this count by the total number of test samples
       
    3. Return the calculated accuracy (a value between 0 and 1)
    """
    # Your implementation here

    predictions = model.predict(test_data)

    correct = 0

    for prediction, true_label in zip(predictions, test_labels):
        if prediction == true_label:
            correct += 1

    accuracy = correct / len(test_labels)

    return accuracy

def main():
    # TODO: Implement the main function to train and evaluate models
    # See README.md for detailed instructions

    data_dir = "data"

    models_dir = "models"

    os.makedirs(models_dir, exist_ok=True)

    configurations = [
        {
            "stopwords_removed": "no",
            "feature_type": "unigrams",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "train_label_file": "train_labels.csv",
            "test_label_file": "test_labels.csv",
            "use_unigrams": True,
            "use_bigrams": False,
            "model_filename": "unigrams_with_stopwords.pkl",
        },
        {
            "stopwords_removed": "no",
            "feature_type": "bigrams",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "train_label_file": "train_labels.csv",
            "test_label_file": "test_labels.csv",
            "use_unigrams": False,
            "use_bigrams": True,
            "model_filename": "bigrams_with_stopwords.pkl",
        },
        {
            "stopwords_removed": "no",
            "feature_type": "unigrams+bigrams",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "train_label_file": "train_labels.csv",
            "test_label_file": "test_labels.csv",
            "use_unigrams": True,
            "use_bigrams": True,
            "model_filename": "unigrams_bigrams_with_stopwords.pkl",
        },
        {
            "stopwords_removed": "yes",
            "feature_type": "unigrams",
            "train_file": "train_ns.csv",
            "test_file": "test_ns.csv",
            "train_label_file": "train_labels.csv",
            "test_label_file": "test_labels.csv",
            "use_unigrams": True,
            "use_bigrams": False,
            "model_filename": "unigrams_without_stopwords.pkl",
        },
        {
            "stopwords_removed": "yes",
            "feature_type": "bigrams",
            "train_file": "train_ns.csv",
            "test_file": "test_ns.csv",
            "train_label_file": "train_labels.csv",
            "test_label_file": "test_labels.csv",
            "use_unigrams": False,
            "use_bigrams": True,
            "model_filename": "bigrams_without_stopwords.pkl",
        },
        {
            "stopwords_removed": "yes",
            "feature_type": "unigrams+bigrams",
            "train_file": "train_ns.csv",
            "test_file": "test_ns.csv",
            "train_label_file": "train_labels.csv",
            "test_label_file": "test_labels.csv",
            "use_unigrams": True,
            "use_bigrams": True,
            "model_filename": "unigrams_bigrams_without_stopwords.pkl",
        }
    ]

    results = []

    for config in configurations:
        print(f"Training model : {config['feature_type']},stopwords removed : {config['stopwords_removed']}")

        train_data = load_data(os.path.join(data_dir, config["train_file"]))
        test_data = load_data(os.path.join(data_dir, config["test_file"]))

        train_labels = load_labels(os.path.join(data_dir, config["train_label_file"]))
        test_labels = load_labels(os.path.join(data_dir, config["test_label_file"]))

        train_texts = [" ".join(tokens) for tokens in train_data]
        test_texts = [" ".join(tokens) for tokens in test_data]

        model = train_model(
            train_texts,
            train_labels,
            use_bigrams=config["use_bigrams"],
            use_unigrams=config["use_unigrams"],
        )

        accuracy = evaluate_model(model, test_texts, test_labels)

        model_path = os.path.join(models_dir, config["model_filename"])
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=4)

        results.append({
            "Stopwords removed": config["stopwords_removed"],
            "text features": config["feature_type"],
            "Accuracy (test set)": accuracy,
        })

        print(f"Accuracy : {accuracy:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)

    print("Training complete")
    print("Models saved in the models directory")
    print("Results saved to results.csv")

if __name__ == "__main__":
    main()