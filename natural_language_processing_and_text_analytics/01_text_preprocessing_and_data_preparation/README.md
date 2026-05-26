[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bTXeQMu6)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23969188)
# Assignment 1: Data Preparation

**Value**: 5% of final grade

## Overview
In this assignment, you will develop a Python script to perform essential data preparation tasks for text analytics. You will tokenize, clean, and split the Amazon reviews corpus into training, validation, and test sets.

## Learning Objectives
- Implement basic text preprocessing techniques
- Create train/validation/test splits for model development
- Practice Python file handling and string manipulation

## Dataset Information
You will be working with the Amazon reviews corpus which contains two classes of consumer product reviews: positive and negative. The dataset is available at:
https://github.com/fuzhenxin/textstyletransferdata/tree/master/sentiment

Before starting the assignment, download the dataset by running:
```bash
# In the assignment root directory
mkdir -p data
curl -o data/pos.txt https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/pos.txt
curl -o data/neg.txt https://raw.githubusercontent.com/fuzhenxin/textstyletransferdata/master/sentiment/neg.txt
```

## Implementation Pipeline

Your implementation should follow this logical sequence of steps:

1. **Data Loading and Labeling**
   - Load positive reviews from pos.txt and assign "positive" labels
   - Load negative reviews from neg.txt and assign "negative" labels
   - Combine into a single dataset of (text, label) pairs

2. **Data Shuffling**
   - Shuffle the combined dataset while maintaining text-label pairs
   - Use a fixed random seed for reproducibility

3. **Tokenization**
   - Process each review text to create tokens
   - Remove special characters as specified
   - Create a version with stopwords preserved

4. **Stopword Removal**
   - Load stopwords from stopwords.txt
   - Create a second version of tokenized texts with stopwords removed

5. **Data Splitting**
   - Split the tokenized texts and their corresponding labels into train/val/test sets
   - Maintain the same split indices for both versions (with and without stopwords)
   - Ensure 80%/10%/10% ratios

6. **File Writing**
   - Write tokenized texts with stopwords to out.csv, train.csv, val.csv, test.csv
   - Write tokenized texts without stopwords to out_ns.csv, train_ns.csv, val_ns.csv, test_ns.csv
   - Write corresponding labels to out_labels.csv, train_labels.csv, val_labels.csv, test_labels.csv

## Requirements

### 1. Environment Setup
Set up a Python virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Loading and Labeling

Your script must:
- Load reviews from both positive and negative files
- Associate each review with the correct sentiment label
- Maintain this association throughout all subsequent processing steps

Example code structure for loading:

```python
def load_data(data_dir):
    """
    Load positive and negative reviews from the given directory.
    
    Args:
        data_dir (str): Path to directory containing pos.txt and neg.txt
        
    Returns:
        list: List of tuples (review_text, label)
    """
    # Load positive and negative reviews with their labels
    # Return a combined dataset
    pass
```

### 3. Data Shuffling

Before any tokenization or splitting, you must:
- Shuffle the combined dataset to ensure random distribution of reviews
- Use a fixed random seed (e.g., 42) for reproducibility
- Maintain the association between each review and its label

Example code structure for shuffling:

```python
def shuffle_data(labeled_data, seed=42):
    """
    Shuffle the labeled data.
    
    Args:
        labeled_data (list): List of (review_text, label) tuples
        seed (int): Random seed for reproducibility
        
    Returns:
        list: Shuffled list of (review_text, label) tuples
    """
    # Shuffle the data while maintaining text-label pairs
    pass
```

### 4. Tokenization and Text Cleaning

After shuffling, process each review by:
- Tokenizing the text into individual words
- Removing special characters: !#$%&()*+/:,;.<=>@[\]^`{|}~\t\n
- Maintaining the original order of words

Example code structure for tokenization:

```python
def tokenize(text):
    """
    Tokenize a text string.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of tokens
    """
    # Tokenize and clean the text
    pass
```

### 5. Stopword Removal

Create a second version of the tokenized texts by:
- Loading the stopwords list from the provided file
- Removing any token that appears in the stopwords list
- Preserving the original version with stopwords intact

Example code structure for stopword removal:

```python
def load_stopwords(stopwords_path):
    """
    Load stopwords from the provided file.
    
    Args:
        stopwords_path (str): Path to the stopwords file
        
    Returns:
        set: Set of stopwords
    """
    # Load stopwords from file
    pass

def remove_stopwords(tokens, stopwords):
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens (list): List of tokens
        stopwords (set): Set of stopwords
        
    Returns:
        list: List of tokens with stopwords removed
    """
    # Remove stopwords from tokens
    pass
```

### 6. Data Splitting

Split the processed data as follows:
- 80% for training
- 10% for validation
- 10% for testing
- Apply the same split to both tokenized versions and their labels
- Maintain alignment between texts and labels

Example code structure for splitting:

```python
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
    # Split data while maintaining alignment between texts and labels
    pass
```

### 7. Output Files

Your script must generate the following files in the `data/` directory:

#### Text Files (with comma-separated tokens):
- `out.csv`: All tokenized sentences with stopwords
- `train.csv`: Training set with stopwords (80%)
- `val.csv`: Validation set with stopwords (10%)
- `test.csv`: Test set with stopwords (10%)
- `out_ns.csv`: All tokenized sentences without stopwords
- `train_ns.csv`: Training set without stopwords (80%)
- `val_ns.csv`: Validation set without stopwords (10%)
- `test_ns.csv`: Test set without stopwords (10%)

#### Label Files (one label per line):
- `out_labels.csv`: Labels for all sentences ("positive" or "negative")
- `train_labels.csv`: Labels for training set
- `val_labels.csv`: Labels for validation set
- `test_labels.csv`: Labels for test set

Each line in the text CSV files should contain the tokens for a sentence, separated by commas. Each line in the label CSV files should contain the corresponding sentiment label ("positive" or "negative").

**Important**: The line number in each text file must correspond to the same line number in its label file. For example, if line 5 in `train.csv` contains a positive review, then line 5 in `train_labels.csv` must contain "positive".

Example code structure for file writing:

```python
def write_to_csv(tokenized_texts, output_file):
    """
    Write tokenized texts to a CSV file.
    
    Args:
        tokenized_texts (list): List of token lists
        output_file (str): Path to output file
    """
    # Write tokenized texts to CSV
    pass

def write_labels_to_csv(labels, output_file):
    """
    Write labels to a CSV file.
    
    Args:
        labels (list): List of labels
        output_file (str): Path to output file
    """
    # Write labels to CSV
    pass
```

### Expected Output Format

#### Example lines for text files (with stopwords):
```
this,product,is,amazing,i,really,love,it,so,much
the,battery,life,on,this,device,is,quite,impressive,it,lasts,all,day
not,worth,the,money,very,disappointed,with,the,quality
```

#### Corresponding lines for text files (without stopwords):
```
product,amazing,really,love,much
battery,life,device,quite,impressive,lasts,day
worth,money,disappointed,quality
```

#### Corresponding lines for label files:
```
positive
positive
negative
```

## Testing Your Implementation
Your implementation should work when run as follows:
```bash
python main.py data/
```

Make sure to test your implementation with the actual dataset before submission. You can also use the provided test script to verify your implementation meets the requirements:

```bash
python test_assignment1.py data/
```

## Submission Instructions

1. Complete the implementation in `main.py`
2. Ensure your implementation generates all required output files correctly
3. Commit and push your changes to your GitHub repository:
   ```bash
   git add .
   git commit -m "completed Assignment 1"
   git push origin main
   ```
4. Verify that your final submission includes:
   - Completed `main.py` script
   - Any additional helper files you created
   - Do NOT include the data files or generated CSV files in your submission

## Verifying Your Submission

After pushing your changes to GitHub, you can verify your submission passed all tests:

1. Go to your GitHub repository page
2. Click on the "Actions" tab at the top of the repository
3. Look for the most recent workflow run (it should be triggered by your push)
4. If all tests have passed, you will see a green checkmark ✅ next to the workflow run
5. You can click on the workflow run to see detailed test results
6. Make sure all test steps show ✅ and there are no ❌ errors
7. At the bottom of the workflow results, you should see "Points 100/100" if all tests passed successfully

If any tests failed, you can examine the error messages, make corrections to your code, and push the changes again to rerun the tests.

## Git Command Help

If you're new to Git or having trouble with these commands:
- `git add .` stages all changed files for commit
- `git commit -m "completed Assignment 1"` saves your changes with a descriptive message
- `git push origin main` uploads your changes to GitHub

You can check if your submission was successful by visiting your GitHub repository page and confirming that your changes appear there.

## Evaluation Criteria

Your assignment will be evaluated based on the following criteria:

- **Code Compliance (10%)**: Use of only authorized built-in Python libraries
- **File Generation and Structure (10%)**: Creation of all required output files with proper naming conventions
- **CSV Format and Tokenization (10%)**: Proper formatting of output files with comma-separated tokens
- **Data Splitting (20%)**: Correct implementation of 80/10/10 train/validation/test split
- **Stopwords Removal (15%)**: Proper removal of stopwords in the designated output files
- **Label File Handling (15%)**: Correct generation and formatting of label files with proper alignment
- **Overall Implementation (20%)**: Complete and correct implementation of the main program functionality, including proper handling of special characters and tokenization

All tests will be run automatically through a testing framework. You can verify your implementation meets the requirements by running:

```bash
python test_assignment1.py data/
```

This test script checks each of the criteria above and provides feedback on whether your implementation passes each test.

**IMPORTANT**: Using unauthorized external libraries will result in significant point deductions. Implement all functionality (tokenization, stopword removal, data splitting) using only Python's built-in libraries as specified in the requirements.


## Academic Integrity
This assignment must be completed individually. You are encouraged to discuss high-level approaches with classmates, but the code you submit must be your own work. Plagiarism and code sharing will be taken seriously.
