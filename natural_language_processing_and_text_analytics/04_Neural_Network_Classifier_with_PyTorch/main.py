#!/usr/bin/env python3
"""
MSE 641 Assignment 4: Neural Network Classifier with PyTorch
-----------------------------------------
Author: Li Xiaohui
Student ID: 21221482

This script implements a fully-connected feed-forward neural network classifier
for sentiment analysis of Amazon reviews using Word2Vec embeddings.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_embeddings(embeddings_path):
    """
    Load Word2Vec embeddings from Assignment 3

    Args:
        embeddings_path: Path to the Word2Vec model file

    Returns:
        word2vec_model: Loaded Word2Vec model
    """
    from gensim.models import Word2Vec

    if embeddings_path.endswith('.pkl'):
        with open(embeddings_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = Word2Vec.load(embeddings_path)

    return model


def load_data_and_labels(data_file, labels_file):
    """
    Load tokenized data and labels from CSV files

    Args:
        data_file: Path to tokenized data file
        labels_file: Path to labels file

    Returns:
        tuple: (processed_data, processed_labels)
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [line.strip().split(',') for line in f if line.strip()]

    label_df = pd.read_csv(labels_file, header=None)
    labels = label_df[0].tolist()

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return data, labels


def create_document_embeddings(tokenized_documents, word2vec_model):
    """
    Convert tokenized documents to document embeddings using Word2Vec

    Args:
        tokenized_documents: List of tokenized documents (list of lists)
        word2vec_model: Trained Word2Vec model

    Returns:
        numpy.ndarray: Document embeddings matrix
    """
    embedding_dim = word2vec_model.vector_size
    doc_embeddings = []

    for doc in tokenized_documents:
        word_vectors = []
        for word in doc:
            if word in word2vec_model.wv:
                word_vectors.append(word2vec_model.wv[word])

        if word_vectors:
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            doc_vector = np.zeros(embedding_dim)

        doc_embeddings.append(doc_vector)

    return np.array(doc_embeddings)


class NeuralNetworkClassifier(nn.Module):
    """
    Fully-connected neural network classifier for sentiment analysis
    """

    def __init__(self, input_size, hidden_size, num_classes, activation='relu', dropout_rate=0.5):
        """
        Initialize the neural network

        Args:
            input_size: Size of input features (Word2Vec embedding dimension)
            hidden_size: Size of hidden layer
            num_classes: Number of output classes
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            dropout_rate: Dropout rate for regularization
        """
        super(NeuralNetworkClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output logits
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, device, num_epochs=100, learning_rate=0.001, l2_reg=0.001):
    """
    Train the neural network model

    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use (cuda or cpu)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        l2_reg: L2 regularization strength

    Returns:
        dict: Training history with losses and accuracies, best validation accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    best_val_acc = 0.0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return {'best_val_acc': best_val_acc}


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data

    Args:
        model: Trained neural network model
        test_loader: DataLoader for test data
        device: Device to use (cuda or cpu)

    Returns:
        float: Test accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def make_sure_dir_exists(dir_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main():
    """
    Main execution function
    """
    print("MSCI 641 Assignment 4: Neural Network Classifier")
    print("=" * 50)

    # Configuration
    EMBEDDING_PATH = "embeddings/word2vec_model.model"
    DATA_DIR = "data"
    MODELS_DIR = "models"

    # Hyperparameters
    HIDDEN_SIZE = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    # Experimental configurations
    experiments = [
        # (activation, l2_reg, dropout_rate)
        ('relu', 0.001, 0.3),
        ('relu', 0.01, 0.3),
        ('relu', 0.001, 0.5),
        ('sigmoid', 0.001, 0.3),
        ('sigmoid', 0.01, 0.3),
        ('tanh', 0.001, 0.3),
        ('tanh', 0.01, 0.3),
    ]

    # Create models directory
    make_sure_dir_exists(MODELS_DIR)

    # Step 1: Load Word2Vec embeddings
    word2vec_model = load_embeddings(EMBEDDING_PATH)
    input_size = word2vec_model.vector_size

    # Step 2: Load data
    train_data, train_labels = load_data_and_labels(f"{DATA_DIR}/train.csv", f"{DATA_DIR}/train_labels.csv")
    val_data, val_labels = load_data_and_labels(f"{DATA_DIR}/val.csv", f"{DATA_DIR}/val_labels.csv")
    test_data, test_labels = load_data_and_labels(f"{DATA_DIR}/test.csv", f"{DATA_DIR}/test_labels.csv")

    # Step 3: Create document embeddings
    train_embeddings = create_document_embeddings(train_data, word2vec_model)
    val_embeddings = create_document_embeddings(val_data, word2vec_model)
    test_embeddings = create_document_embeddings(test_data, word2vec_model)

    # Step 4: Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(train_embeddings), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(torch.FloatTensor(val_embeddings), torch.LongTensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset = TensorDataset(torch.FloatTensor(test_embeddings), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 5: Run experiments
    results = []
    best_models = {}
    overall_best_acc = 0
    overall_best_model = None

    for activation, l2_reg, dropout in experiments:
        print(f"\nTraining model: activation={activation}, L2={l2_reg}, dropout={dropout}")

        model = NeuralNetworkClassifier(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_classes=2,
            activation=activation,
            dropout_rate=dropout
        )

        model = model.to(device)
        history = train_model(model, train_loader, val_loader, device,
                             num_epochs=NUM_EPOCHS,
                             learning_rate=LEARNING_RATE,
                             l2_reg=l2_reg)

        results.append({
            'Activation Function': activation,
            'L2 Regularization': l2_reg,
            'Dropout Rate': dropout,
            'Validation Accuracy': history['best_val_acc']
        })

        if activation not in best_models or history['best_val_acc'] > best_models[activation]['acc']:
            best_models[activation] = {
                'model': model,
                'acc': history['best_val_acc'],
                'l2': l2_reg,
                'dropout': dropout
            }

        if history['best_val_acc'] > overall_best_acc:
            overall_best_acc = history['best_val_acc']
            overall_best_model = model

    # Step 6: Save hyperparameter tuning results
    pd.DataFrame(results).to_csv('hyperparameter_tuning.csv', index=False)

    # Step 7: Evaluate best models on test set
    final_results = []
    for activation in ['relu', 'sigmoid', 'tanh']:
        info = best_models[activation]
        test_acc = evaluate_model(info['model'], test_loader, device)
        final_results.append({
            'Activation Function': activation,
            'L2 Regularization': info['l2'],
            'Dropout Rate': info['dropout'],
            'Validation Accuracy': info['acc'],
            'Test Accuracy': test_acc
        })
        torch.save(info['model'].state_dict(), f"{MODELS_DIR}/{activation}.pth")

    pd.DataFrame(final_results).to_csv('results.csv', index=False)

    # Step 8: Save overall best model
    torch.save(overall_best_model.state_dict(), f"{MODELS_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
