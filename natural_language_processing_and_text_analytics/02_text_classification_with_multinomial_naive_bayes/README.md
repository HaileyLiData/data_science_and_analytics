# Multinomial Naive Bayes Text Classification

Text classification workflow using Multinomial Naive Bayes for sentiment analysis on Amazon review data.

The project evaluates multiple feature engineering strategies, including unigram, bigram, and combined unigram-bigram representations, with and without stopword removal.

---

## Module

02_Text_Classification_with_Multinomial_Naive_Bayes

---

## Project Overview

This project builds a complete text classification pipeline using Multinomial Naive Bayes.

The workflow uses the preprocessed datasets generated in Module 01 and compares different feature extraction strategies for sentiment classification.

The project focuses on:

* text vectorization
* n-gram feature engineering
* sentiment classification
* model evaluation
* model persistence
* comparative NLP experimentation

---

## Classification Workflow

### Step 1 — Load Preprocessed Datasets

Loading training and testing datasets generated during preprocessing.

Input datasets include:

* train.csv
* test.csv
* train_ns.csv
* test_ns.csv
* train_labels.csv
* test_labels.csv

---

### Step 2 — Configure Feature Representation

Three feature configurations are evaluated:

#### Unigrams

Single-word features.

Example:

good

movie

excellent

---

#### Bigrams

Two-word sequence features.

Example:

good movie

movie excellent

---

#### Unigrams + Bigrams

Combined feature space using both single words and word pairs.

---

### Step 3 — Vectorize Text Data

Text documents are transformed into numerical feature vectors using:

CountVectorizer

The vectorizer converts tokenized text into document-term matrices suitable for machine learning models.

---

### Step 4 — Train Multinomial Naive Bayes Models

Training sentiment classifiers using:

MultinomialNB

The classifier learns relationships between word frequencies and sentiment labels.

---

### Step 5 — Evaluate Model Performance

Model predictions are compared against true labels.

Evaluation metric:

Accuracy

Accuracy = Correct Predictions / Total Predictions

---

### Step 6 — Save Trained Models

Each trained model is serialized using pickle for future reuse.

Generated model files:

* unigrams_with_stopwords.pkl
* bigrams_with_stopwords.pkl
* unigrams_bigrams_with_stopwords.pkl
* unigrams_without_stopwords.pkl
* bigrams_without_stopwords.pkl
* unigrams_bigrams_without_stopwords.pkl

---

## Experimental Design

The project evaluates six model configurations.

| Stopwords Removed | Features           |
| ----------------- | ------------------ |
| No                | Unigrams           |
| No                | Bigrams            |
| No                | Unigrams + Bigrams |
| Yes               | Unigrams           |
| Yes               | Bigrams            |
| Yes               | Unigrams + Bigrams |

This allows comparison of:

* stopword effects
* unigram effectiveness
* bigram effectiveness
* combined feature effectiveness

---

## Technologies Used

* Python
* pandas
* numpy
* scikit-learn
* pickle

---

## NLP Concepts Demonstrated

This project demonstrates:

* text classification
* supervised learning
* Multinomial Naive Bayes
* Bag-of-Words representation
* CountVectorizer
* unigram features
* bigram features
* stopword analysis
* sentiment analysis
* model evaluation
* model serialization

---

## Academic Context

Developed as part of graduate-level coursework in:

MSE 641 — Text Analytics / Natural Language Processing

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
```

## Repository Notes

This project is organized as a portfolio-style NLP classification module focused on practical sentiment analysis workflows and comparative feature engineering experiments.

Large datasets are excluded from version control whenever possible.
