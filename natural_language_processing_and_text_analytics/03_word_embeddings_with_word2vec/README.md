# Word Embeddings with Word2Vec

Word embedding workflow using Word2Vec for learning semantic word representations from Amazon review data.

The project trains a Word2Vec model using tokenized review text and analyzes semantic relationships between words through vector similarity.

---

## Module

03_Word_Embeddings_with_Word2Vec

---

## Project Overview

This project builds a complete Word2Vec training pipeline using the Gensim library.

The workflow uses the preprocessed datasets generated in Module 01 and learns distributed word representations from Amazon review data.

The project focuses on:

* word embeddings
* distributional semantics
* Word2Vec training
* vocabulary construction
* similarity analysis
* vector representations
* semantic relationship discovery

---

## Word2Vec Workflow

### Step 1 — Load Preprocessed Training Data

Loading tokenized review data generated during preprocessing.

Input dataset used:

* train.csv

This project uses the full tokenized training dataset generated during the text preprocessing stage. The version with stopwords retained was selected for Word2Vec training in order to preserve contextual information and semantic relationships between words.

---

### Step 2 — Build Vocabulary

Word2Vec scans the training corpus and constructs a vocabulary.

Vocabulary construction includes:

* word frequency counting
* filtering low-frequency words
* vocabulary indexing

Words appearing fewer than the specified minimum count are excluded.

---

### Step 3 — Train Word Embeddings

Training distributed word representations using:

Word2Vec (Gensim)

Recommended configuration:

* vector_size = 100
* window = 5
* min_count = 5
* epochs = 10
* sg = 1 (Skip-Gram)

The Skip-Gram model learns word meanings by predicting surrounding context words.

---

### Step 4 — Generate Word Vectors

After training, every word in the vocabulary is represented as a dense numerical vector.

Example:

good → [0.12, -0.45, 0.87, ...]

excellent → [0.15, -0.41, 0.82, ...]

These vectors capture semantic relationships between words.

---

### Step 5 — Analyze Semantic Similarity

The trained model is used to identify words with similar meanings.

Target words:

* good
* bad

For each target word, the model retrieves:

* top 20 most similar words
* similarity scores

This demonstrates how Word2Vec captures semantic patterns from text.

---

### Step 6 — Save Model and Results

Generated output files:

* word2vec_model.model
* word_similarities.json
* analysis.txt

The trained model can be reused for future NLP experiments.

---

## Experimental Design

The project evaluates semantic relationships learned from Amazon review data.

Analysis focuses on:

| Target Word | Purpose                        |
| ----------- | ------------------------------ |
| good        | Positive semantic neighborhood |
| bad         | Negative semantic neighborhood |

The experiment investigates whether Word2Vec naturally captures sentiment-related patterns through contextual learning.

---

## Technologies Used

* Python
* pandas
* numpy
* gensim
* json

---

## NLP Concepts Demonstrated

This project demonstrates:

* word embeddings
* Word2Vec
* distributed representations
* semantic similarity
* vector space models
* Skip-Gram
* context learning
* vocabulary construction
* unsupervised learning
* sentiment-related semantic patterns

---

## Key NLP Evolution

This project represents an important stage in the evolution of Natural Language Processing:

Bag-of-Words

↓

TF-IDF

↓

Word2Vec

↓

Transformer

↓

BERT

↓

GPT

Word2Vec was one of the first successful approaches to learning meaningful word representations from large text corpora.

---

## Academic Context

Developed as part of graduate-level coursework in:

MSE 641 — Text Analytics / Natural Language Processing

University of Waterloo

---

## Repository Structure

```text
03_word_embeddings_with_word2vec
│
├── README.md
├── main.py
├── requirements.txt
├── analysis.txt
├── word_similarities.json
│
└── word2vec_model.model
```

---

## Repository Notes

This project is organized as a portfolio-style NLP module focused on learning distributed word representations and exploring semantic relationships through Word2Vec.

Large datasets are excluded from version control whenever possible.

The model is trained on preprocessed Amazon review data generated in Module 01.
