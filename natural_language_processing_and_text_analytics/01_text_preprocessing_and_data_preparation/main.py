#!/usr/bin/env python3
"""
MSE 641 Assignment 1: Data Preparation
-----------------------------------------
This script performs tokenization, cleaning, and data splitting on the Amazon reviews corpus.

Author: Li Xiaohui
Student ID: 21221482
"""

import argparse
import os
import random
import re
import string


def load_data(data_dir):
    """
    Load positive and negative reviews from the given directory.
    
    Args:
        data_dir (str): Path to directory containing pos.txt and neg.txt
        
    Returns:
        list: List of tuples (review_text, label)
    """
    # TODO: Implement this function
    # Load positive reviews from pos.txt and associate each with the label "positive"
    # Load negative reviews from neg.txt and associate each with the label "negative"
    # Return a combined list of (review_text, label) tuples

    labeled_data = []

    pos_path = os.path.join(data_dir, "pos.txt")
    neg_path = os.path.join(data_dir, "neg.txt")

    with open(pos_path, "r", encoding="utf-8") as f:
        for line in f:
            labeled_data.append((line.strip(), "positive"))

    with open(neg_path, "r", encoding="utf-8") as f:
        for line in f:
            labeled_data.append((line.strip(), "negative"))

    return labeled_data


def shuffle_data(labeled_data, seed=42):
    """
    Shuffle the labeled data.
    
    Args:
        labeled_data (list): List of (review_text, label) tuples
        seed (int): Random seed for reproducibility
        
    Returns:
        list: Shuffled list of (review_text, label) tuples
    """
    # TODO: Implement this function
    # Shuffle the data while maintaining text-label pairs
    # Use the provided seed for reproducibility

    random.seed(seed)

    shuffled_data = labeled_data.copy()

    random.shuffle(shuffled_data)

    return shuffled_data


def tokenize(text):
    """
    Tokenize a text string.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of tokens
    """
    # TODO: Implement this function
    # Tokenize text into individual words
    # Remove specified special characters: !#$%&()*+/:,;.<=>@[\]^`{|}~\t\n

    special_chars = r'[!#$%&()*+/:,;.<=>@\[\]^`{|}~\t\n]'

    clean_text = re.sub(special_chars, ' ', text)

    clean_text = clean_text.lower()

    clean_text = re.sub(r'(.)\1{2,}', r'\1\1', clean_text)

    tokens = clean_text.split()

    return tokens


def load_stopwords(stopwords_path):
    """
    Load stopwords from the provided file.
    
    Args:
        stopwords_path (str): Path to the stopwords file
        
    Returns:
        set: Set of stopwords
    """
    # TODO: Implement this function
    # Load and return stopwords from the provided file path
    # Handle file not found errors gracefully

    try:

        stopwords = set()

        with open(stopwords_path, "r", encoding="utf-8") as f:

            for line in f:
                word = line.strip()

                stopwords.add(word)

        return stopwords

    except FileNotFoundError:

        print("Stopwords file not found.")

        return set()


def remove_stopwords(tokens, stopwords):
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens (list): List of tokens
        stopwords (set): Set of stopwords
        
    Returns:
        list: List of tokens with stopwords removed
    """
    # TODO: Implement this function
    # Remove any token that appears in the stopwords set

    filtered_tokens = []

    for token in tokens:

        if token not in stopwords:
            filtered_tokens.append(token)

    return filtered_tokens


def split_data(tokenized_texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split data into training, validation, and test sets.
    
    Args:
        tokenized_texts (list): List of tokenized texts
        labels (list): List of corresponding labels
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    # TODO: Implement this function
    # Ensure the split is randomly sampled but reproducible
    # Check that ratios sum to 1.0
    # Maintain alignment between texts and their labels

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Ratios must sum to 1.0")

    random.seed(seed)

    combined = list(zip(tokenized_texts, labels))

    random.shuffle(combined)

    tokenized_texts, labels = zip(*combined)

    total_size = len(tokenized_texts)

    train_end = int(total_size * train_ratio)

    val_end = train_end + int(total_size * val_ratio)

    train_texts = tokenized_texts[:train_end]
    val_texts = tokenized_texts[train_end:val_end]
    test_texts = tokenized_texts[val_end:]

    train_labels = labels[:train_end]
    val_labels = labels[train_end:val_end]
    test_labels = labels[val_end:]

    return (
        train_texts,
        val_texts,
        test_texts,
        train_labels,
        val_labels,
        test_labels
    )


def write_to_csv(tokenized_texts, output_file):
    """
    Write tokenized texts to a CSV file.
    
    Args:
        tokenized_texts (list): List of token lists
        output_file (str): Path to output file
    """
    # TODO: Implement this function
    # Write each list of tokens as a comma-separated string

    with open(output_file, "w", encoding="utf-8") as f:

        for tokens in tokenized_texts:

            line = ",".join(tokens)

            f.write(line + "\n")


def write_labels_to_csv(labels, output_file):
    """
    Write labels to a CSV file.
    
    Args:
        labels (list): List of labels
        output_file (str): Path to output file
    """
    # TODO: Implement this function
    # Write each label on a separate line

    with open(output_file, "w", encoding="utf-8") as f:

        for label in labels:

            f.write(label + "\n")


def main():
    parser = argparse.ArgumentParser(description='MSCI 641 Assignment 1: Data Preparation')
    parser.add_argument('data_dir', type=str, help='Path to directory containing pos.txt and neg.txt')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # Check if data files exist
    pos_path = os.path.join(data_dir, 'pos.txt')
    neg_path = os.path.join(data_dir, 'neg.txt')
    stopwords_path = os.path.join(data_dir, 'stopwords.txt')
    
    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        print(f"Error: pos.txt or neg.txt not found in {data_dir}")
        return
    
    if not os.path.exists(stopwords_path):
        print(f"Error: stopwords.txt not found in {data_dir}")
        return
    
    # TODO: Implement the main workflow:
    # 1. Load data (combine positive and negative reviews with their labels)
    # labeled_data = load_data(data_dir)
    
    # 2. Shuffle the data
    # shuffled_data = shuffle_data(labeled_data)
    
    # 3. Separate texts and labels
    # texts = [item[0] for item in shuffled_data]
    # labels = [item[1] for item in shuffled_data]
    
    # 4. Tokenize the texts
    # tokenized_texts = [tokenize(text) for text in texts]
    
    # 5. Load stopwords
    # stopwords = load_stopwords(stopwords_path)
    
    # 6. Create version without stopwords
    # tokenized_texts_ns = [remove_stopwords(tokens, stopwords) for tokens in tokenized_texts]
    
    # 7. Split the data into train/val/test sets
    # train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(tokenized_texts, labels)
    # train_texts_ns, val_texts_ns, test_texts_ns, _, _, _ = split_data(tokenized_texts_ns, labels)
    
    # 8. Write tokenized texts and labels to CSV files
    # write_to_csv(tokenized_texts, os.path.join(data_dir, 'out.csv'))
    # write_to_csv(train_texts, os.path.join(data_dir, 'train.csv'))
    # write_to_csv(val_texts, os.path.join(data_dir, 'val.csv'))
    # write_to_csv(test_texts, os.path.join(data_dir, 'test.csv'))
    
    # write_to_csv(tokenized_texts_ns, os.path.join(data_dir, 'out_ns.csv'))
    # write_to_csv(train_texts_ns, os.path.join(data_dir, 'train_ns.csv'))
    # write_to_csv(val_texts_ns, os.path.join(data_dir, 'val_ns.csv'))
    # write_to_csv(test_texts_ns, os.path.join(data_dir, 'test_ns.csv'))
    
    # write_labels_to_csv(labels, os.path.join(data_dir, 'out_labels.csv'))
    # write_labels_to_csv(train_labels, os.path.join(data_dir, 'train_labels.csv'))
    # write_labels_to_csv(val_labels, os.path.join(data_dir, 'val_labels.csv'))
    # write_labels_to_csv(test_labels, os.path.join(data_dir, 'test_labels.csv'))


    labeled_data = load_data(data_dir)

    shuffled_data = shuffle_data(labeled_data)

    texts = [item[0] for item in shuffled_data]
    labels = [item[1] for item in shuffled_data]

    tokenized_texts = [tokenize(text) for text in texts]

    stopwords = load_stopwords(stopwords_path)

    tokenized_texts_ns = [remove_stopwords(tokens, stopwords) for tokens in tokenized_texts]

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(tokenized_texts, labels)
    train_texts_ns, val_texts_ns, test_texts_ns, _, _, _ = split_data(tokenized_texts_ns, labels)

    write_to_csv(tokenized_texts, os.path.join(data_dir, 'out.csv'))
    write_to_csv(train_texts, os.path.join(data_dir, 'train.csv'))
    write_to_csv(val_texts, os.path.join(data_dir, 'val.csv'))
    write_to_csv(test_texts, os.path.join(data_dir, 'test.csv'))

    write_to_csv(tokenized_texts_ns, os.path.join(data_dir, 'out_ns.csv'))
    write_to_csv(train_texts_ns, os.path.join(data_dir, 'train_ns.csv'))
    write_to_csv(val_texts_ns, os.path.join(data_dir, 'val_ns.csv'))
    write_to_csv(test_texts_ns, os.path.join(data_dir, 'test_ns.csv'))

    write_labels_to_csv(labels, os.path.join(data_dir, 'out_labels.csv'))
    write_labels_to_csv(train_labels, os.path.join(data_dir, 'train_labels.csv'))
    write_labels_to_csv(val_labels, os.path.join(data_dir, 'val_labels.csv'))
    write_labels_to_csv(test_labels, os.path.join(data_dir, 'test_labels.csv'))

    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()