# Amazon Review Sentiment Dataset Preprocessing Pipeline

Natural Language Processing (NLP) preprocessing workflow for transforming raw Amazon review text into structured, machine-learning-ready datasets.

The project focuses on preprocessing, cleaning, tokenization, stopword filtering, dataset splitting, and CSV dataset generation for downstream NLP and sentiment analysis tasks.

---

## Module

01_Text_Preprocessing_and_Data_Preparation

---

## Project Overview

This project builds a complete NLP preprocessing workflow using Python. The implementation follows a modular pipeline design, where each function handles one specific preprocessing step and the `main()` function connects all components into a complete workflow.

The workflow prepares raw textual data for downstream NLP and machine learning tasks through structured preprocessing, dataset construction, and reproducible data splitting.

---

## NLP Preprocessing Pipeline

### Step 1 — Load Data
- loading positive and negative review datasets
- assigning sentiment labels
- constructing labeled supervised-learning datasets

### Step 2 — Shuffle the Data
- randomized dataset shuffling
- preserving text-label alignment
- reproducible randomization using fixed random seeds

### Step 3 — Separate Texts and Labels
- separating review texts and sentiment labels
- preparing machine learning feature-target structures
- extracting texts and labels using list comprehension

### Step 4 — Tokenize the Texts
- regex-based text cleaning
- removing specified special characters
- lowercase normalization
- repeated-character normalization
- word-level tokenization
- whitespace-based token splitting

### Step 5 — Load Stopwords
- loading stopwords from external files
- set-based stopword storage
- efficient stopword lookup

### Step 6 — Create Version Without Stopwords
- stopword removal
- generating no-stopword dataset versions
- reducing high-frequency functional vocabulary

### Step 7 — Split the Data into Train / Validation / Test Sets
- train / validation / test splitting
- reproducible dataset partitioning
- preserving text-label alignment across splits
- generating parallel stopword-filtered datasets

### Step 8 — Write Tokenized Texts and Labels to CSV Files
- CSV dataset generation
- token serialization
- label file generation
- machine-learning-ready output construction

---

## Dataset

The workflow uses an Amazon review corpus containing two sentiment classes:

- positive reviews
- negative reviews

Expected input files:

```text
data/pos.txt
data/neg.txt
data/stopwords.txt
```

Each line in the review files represents one review sample.

---

## Text Processing Rules

The preprocessing pipeline includes several normalization and cleaning strategies:

### Regex-Based Cleaning
Specified special characters are removed using regular expressions.

### Lowercase Normalization
All review text is converted into lowercase format to improve token consistency.

Examples:

```text
Amazing → amazing
GOOD → good
```

### Repeated Character Normalization
Excessive repeated characters are normalized using regex-based replacement.

Examples:

```text
sooooo → soo
yesssss → yess
```

This reduces unnecessary spelling variation while preserving partial emotional emphasis.

### Word-Level Tokenization
The cleaned text is tokenized into word-level tokens using whitespace splitting.

Example:

```text
this product is very good
```

becomes:

```python
["this", "product", "is", "very", "good"]
```

---

## Output Files

The script generates the following output files:

```text
out.csv              all tokenized reviews
train.csv            training text data
val.csv              validation text data
test.csv             test text data

out_ns.csv           all tokenized reviews with stopwords removed
train_ns.csv         training text data with stopwords removed
val_ns.csv           validation text data with stopwords removed
test_ns.csv          test text data with stopwords removed

out_labels.csv       labels for all reviews
train_labels.csv     labels for training data
val_labels.csv       labels for validation data
test_labels.csv      labels for test data
```

---

## How to Run

Run the preprocessing pipeline:

```bash
python main.py data/
```

Verify the implementation using the provided test script:

```bash
python test_assignment1.py data/
```

---

## Technologies Used

- Python
- argparse
- os
- random
- re (Regular Expressions)
- CSV file generation
- command-line workflow execution

---

## NLP Engineering Concepts Demonstrated

This project demonstrates:

- NLP preprocessing pipeline construction
- regex-based text normalization
- tokenization workflows
- stopword filtering
- dataset engineering
- train / validation / test splitting
- supervised learning dataset preparation
- text-label alignment preservation
- modular Python function design
- reproducible preprocessing workflows

---

## Academic Context

Developed as part of graduate-level coursework in:

**MSE 641 — Text Analytics / Natural Language Processing**  
University of Waterloo

---

## Repository Structure

```text
02_text_classification_with_multinomial_naive_bayes
│
├── README.md
├── main.py
├── requirements.txt
│
└── models
    ├── unigrams_with_stopwords.pkl
    ├── bigrams_with_stopwords.pkl
    ├── unigrams_bigrams_with_stopwords.pkl
    ├── unigrams_without_stopwords.pkl
    ├── bigrams_without_stopwords.pkl
    └── unigrams_bigrams_without_stopwords.pkl

---

## Repository Notes

This project is organized as a portfolio-style NLP preprocessing module focused on practical text preprocessing and dataset engineering workflows.

Generated CSV files and large raw datasets are excluded from version control whenever possible.
