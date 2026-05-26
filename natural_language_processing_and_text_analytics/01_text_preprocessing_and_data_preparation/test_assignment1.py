#!/usr/bin/env python3
"""
Test script for MSE 641 Assignment 1.
This script tests if the assignment output files are generated correctly.
Supports selective testing via command-line arguments for autograding.
"""

import os
import sys
import argparse
from pathlib import Path


def check_file_exists(file_path):
    """Check if a file exists and return True/False."""
    exists = os.path.isfile(file_path)
    status = "✅" if exists else "❌"
    print(f"{status} {file_path}")
    return exists

def check_file_not_empty(file_path):
    """Check if a file is not empty and return True/False."""
    if not os.path.isfile(file_path):
        print(f"❌ {file_path} does not exist")
        return False
    
    size = os.path.getsize(file_path)
    status = "✅" if size > 0 else "❌"
    print(f"{status} {file_path} is {'not empty' if size > 0 else 'empty'}")
    return size > 0

def check_csv_format(file_path, max_lines_to_check=10):
    """Check if a file follows the CSV format with comma-separated tokens."""
    if not os.path.isfile(file_path):
        print(f"❌ {file_path} does not exist")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()[:max_lines_to_check]]
        
        if not lines:
            print(f"❌ {file_path} is empty")
            return False
        
        all_valid = True
        for i, line in enumerate(lines):
            # Check if line contains at least one comma
            if ',' not in line and len(line) > 0:
                print(f"❌ Line {i+1} in {file_path} is not comma-separated: {line[:50]}...")
                all_valid = False
        
        status = "✅" if all_valid else "❌"
        print(f"{status} {file_path} format {'is valid' if all_valid else 'has errors'}")
        return all_valid
    
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")
        return False

def check_stopwords_usage(data_dir):
    """Check if stopwords are correctly removed in the *_ns.csv files."""
    if not os.path.isfile(data_dir / 'stopwords.txt'):
        print("❌ stopwords.txt file not found in the data directory")
        return False
    
    try:
        # Load stopwords
        with open(data_dir / 'stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f if line.strip()}
        
        if len(stopwords) < 50:  # Basic check that we have a reasonable number of stopwords
            print("❌ stopwords.txt contains too few stopwords")
            return False
        
        # Check a few sample files to ensure stopwords are removed
        standard_file = data_dir / 'out.csv'
        no_stopwords_file = data_dir / 'out_ns.csv'
        
        if not (os.path.isfile(standard_file) and os.path.isfile(no_stopwords_file)):
            print("❌ Cannot check stopwords removal: missing output files")
            return False
        
        # Read the first few lines of each file
        with open(standard_file, 'r', encoding='utf-8') as f_std:
            std_lines = [line.strip() for line in f_std.readlines()[:5]]
        
        with open(no_stopwords_file, 'r', encoding='utf-8') as f_ns:
            ns_lines = [line.strip() for line in f_ns.readlines()[:5]]
        
        # Ensure lines correspond to each other
        if len(std_lines) != len(ns_lines):
            print("❌ Standard and no-stopwords files have different numbers of lines")
            return False
        
        # Check if stopwords are properly removed
        proper_removal = True
        for i, (std_line, ns_line) in enumerate(zip(std_lines, ns_lines)):
            std_tokens = std_line.split(',')
            ns_tokens = ns_line.split(',')
            
            # Check if no-stopwords line has fewer tokens
            if len(ns_tokens) >= len(std_tokens):
                print(f"❌ Line {i+1}: No-stopwords line should have fewer tokens than standard line")
                proper_removal = False
                continue
            
            # Check if all tokens in no-stopwords line are present in standard line
            for token in ns_tokens:
                if token not in std_tokens:
                    print(f"❌ Line {i+1}: Token '{token}' in no-stopwords line not found in standard line")
                    proper_removal = False
                    break
            
            # Check if some stopwords are removed
            removed_tokens = set(std_tokens) - set(ns_tokens)
            if not any(token.lower() in stopwords for token in removed_tokens):
                print(f"❌ Line {i+1}: No stopwords appear to have been removed")
                proper_removal = False
        
        status = "✅" if proper_removal else "❌"
        print(f"{status} Stopwords removal {'is valid' if proper_removal else 'has errors'}")
        return proper_removal
        
    except Exception as e:
        print(f"❌ Error checking stopwords usage: {e}")
        return False
    
def check_data_split_sizes(train_file, val_file, test_file):
    """Check if the data is split according to the 80/10/10 ratio."""
    files_exist = all(os.path.isfile(f) for f in [train_file, val_file, test_file])
    if not files_exist:
        print("❌ One or more split files do not exist")
        return False
    
    try:
        train_lines = sum(1 for _ in open(train_file, 'r', encoding='utf-8'))
        val_lines = sum(1 for _ in open(val_file, 'r', encoding='utf-8'))
        test_lines = sum(1 for _ in open(test_file, 'r', encoding='utf-8'))
        
        total = train_lines + val_lines + test_lines
        train_ratio = train_lines / total if total > 0 else 0
        val_ratio = val_lines / total if total > 0 else 0
        test_ratio = test_lines / total if total > 0 else 0
        
        # Allow for small deviations due to rounding
        train_ok = 0.75 <= train_ratio <= 0.85
        val_ok = 0.05 <= val_ratio <= 0.15
        test_ok = 0.05 <= test_ratio <= 0.15
        
        print(f"Split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
        
        status = "✅" if (train_ok and val_ok and test_ok) else "❌"
        print(f"{status} Data split {'is valid' if (train_ok and val_ok and test_ok) else 'is not valid'}")
        
        return train_ok and val_ok and test_ok
    
    except Exception as e:
        print(f"❌ Error checking split sizes: {e}")
        return False

def check_unauthorized_imports(script_path):
    """Check if the script imports any unauthorized libraries."""
    unauthorized_libs = [
        'nltk', 'sklearn', 'spacy', 'pandas', 'numpy', 'gensim', 
        'textblob', 'transformers', 'huggingface', 'torch', 'tensorflow',
        'keras', 'scipy', 'matplotlib', 'seaborn'
    ]
    
    allowed_libs = [
        'argparse', 'os', 'random', 're', 'string', 'sys', 'pathlib'
    ]
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for import statements
        import_lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
        
        # Look for unauthorized imports
        unauthorized_found = []
        for line in import_lines:
            for lib in unauthorized_libs:
                # Check for either "import lib" or "from lib import..."
                if f"import {lib}" in line or f"from {lib}" in line:
                    unauthorized_found.append(lib)
        
        if unauthorized_found:
            print("❌ Unauthorized libraries detected:")
            for lib in unauthorized_found:
                print(f"  - {lib}")
            print("Only the following libraries are allowed:")
            for lib in allowed_libs:
                print(f"  - {lib}")
            return False
        
        print("✅ No unauthorized libraries detected")
        return True
        
    except Exception as e:
        print(f"❌ Error checking imports: {e}")
        return False

def check_label_file(file_path):
    """Check if a label file exists and contains valid labels."""
    if not os.path.isfile(file_path):
        print(f"❌ {file_path} does not exist")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f]
        
        if not labels:
            print(f"❌ {file_path} is empty")
            return False
        
        valid_labels = {"positive", "negative"}
        all_valid = True
        for i, label in enumerate(labels):
            if label not in valid_labels:
                print(f"❌ Line {i+1} in {file_path} contains invalid label: {label}")
                all_valid = False
        
        status = "✅" if all_valid else "❌"
        print(f"{status} {file_path} {'contains valid labels' if all_valid else 'has invalid labels'}")
        return all_valid
    
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")
        return False


def check_label_alignment(text_file, label_file):
    """Check if the number of lines in text and label files match."""
    if not (os.path.isfile(text_file) and os.path.isfile(label_file)):
        print(f"❌ Cannot check alignment: {text_file} or {label_file} does not exist")
        return False
    
    try:
        # Count the actual number of lines, including empty ones
        text_line_count = sum(1 for _ in open(text_file, 'r', encoding='utf-8'))
        label_line_count = sum(1 for _ in open(label_file, 'r', encoding='utf-8'))
        
        if text_line_count != label_line_count:
            print(f"❌ Alignment error: {text_file} has {text_line_count} lines but {label_file} has {label_line_count} lines")
            return False
        
        print(f"✅ {text_file} and {label_file} are properly aligned with {text_line_count} lines each")
        return True
    
    except Exception as e:
        print(f"❌ Error checking alignment: {e}")
        return False

def check_label_distribution(label_file):
    """Check if the label distribution is reasonably balanced."""
    if not os.path.isfile(label_file):
        print(f"❌ {label_file} does not exist")
        return False
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        
        if not labels:
            print(f"❌ {label_file} is empty")
            return False
        
        pos_count = sum(1 for label in labels if label == "positive")
        neg_count = sum(1 for label in labels if label == "negative")
        total = len(labels)
        
        pos_ratio = pos_count / total if total > 0 else 0
        neg_ratio = neg_count / total if total > 0 else 0
        
        print(f"Label distribution in {label_file}: Positive={pos_ratio:.2f}, Negative={neg_ratio:.2f}")
        
        # Check if distribution is severely imbalanced (less than 25% of either class)
        balanced = pos_ratio >= 0.25 and neg_ratio >= 0.25
        
        status = "✅" if balanced else "❌"
        print(f"{status} Label distribution {'is reasonably balanced' if balanced else 'is severely imbalanced'}")
        
        return balanced
    
    except Exception as e:
        print(f"❌ Error checking label distribution: {e}")
        return False


def run_label_tests(data_dir):
    """Test if label files exist, contain valid labels, and are properly aligned with text files."""
    print("\n----- Testing label files -----")
    
    # Define pairs of text and label files to check
    file_pairs = [
        (data_dir / 'out.csv', data_dir / 'out_labels.csv'),
        (data_dir / 'train.csv', data_dir / 'train_labels.csv'),
        (data_dir / 'val.csv', data_dir / 'val_labels.csv'),
        (data_dir / 'test.csv', data_dir / 'test_labels.csv'),
        (data_dir / 'out_ns.csv', data_dir / 'out_labels.csv'),  # Same labels file for NS version
        (data_dir / 'train_ns.csv', data_dir / 'train_labels.csv'),
        (data_dir / 'val_ns.csv', data_dir / 'val_labels.csv'),
        (data_dir / 'test_ns.csv', data_dir / 'test_labels.csv')
    ]
    
    # Check if label files exist and contain valid labels
    label_files = {
        data_dir / 'out_labels.csv',
        data_dir / 'train_labels.csv',
        data_dir / 'val_labels.csv',
        data_dir / 'test_labels.csv'
    }
    
    label_files_ok = all(check_label_file(file) for file in label_files)
    
    # Check alignment between text and label files
    alignment_ok = all(check_label_alignment(text_file, label_file) for text_file, label_file in file_pairs)
    
    # Check label distribution in train/val/test sets
    distribution_ok = all(check_label_distribution(file) for file in [
        data_dir / 'train_labels.csv',
        data_dir / 'val_labels.csv',
        data_dir / 'test_labels.csv'
    ])
    
    result = label_files_ok and alignment_ok and distribution_ok
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\nLabel files are correct: {status}")
    
    return result

def run_import_check(script_path):
    """Test if the script uses only authorized libraries."""
    print("\n----- Testing for unauthorized libraries -----")
    
    import_check_ok = check_unauthorized_imports(script_path)
    
    status = "✅ PASS" if import_check_ok else "❌ FAIL"
    print(f"\nLibrary usage check: {status}")
    
    return import_check_ok

def run_file_existence_tests(data_dir):
    """Test if all required output files exist."""
    output_files = [
        data_dir / 'out.csv',
        data_dir / 'train.csv',
        data_dir / 'val.csv',
        data_dir / 'test.csv',
        data_dir / 'out_ns.csv',
        data_dir / 'train_ns.csv',
        data_dir / 'val_ns.csv',
        data_dir / 'test_ns.csv'
    ]
    
    print("\n----- Testing if files exist -----")
    all_files_exist = all(check_file_exists(file) for file in output_files)
    all_files_not_empty = all(check_file_not_empty(file) for file in output_files)
    
    result = all_files_exist and all_files_not_empty
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\nAll files exist: {status}")
    
    return result

def run_format_tests(data_dir):
    """Test if files follow the required CSV format."""
    output_files = [
        data_dir / 'out.csv',
        data_dir / 'train.csv',
        data_dir / 'val.csv',
        data_dir / 'test.csv',
        data_dir / 'out_ns.csv',
        data_dir / 'train_ns.csv',
        data_dir / 'val_ns.csv',
        data_dir / 'test_ns.csv'
    ]
    
    print("\n----- Testing file formats -----")
    all_files_valid_format = all(check_csv_format(file) for file in output_files)
    
    status = "✅ PASS" if all_files_valid_format else "❌ FAIL"
    print(f"\nAll files follow CSV format: {status}")
    
    return all_files_valid_format

def run_split_tests(data_dir):
    """Test if data split follows 80/10/10 ratio."""
    print("\n----- Testing data splits -----")
    
    with_stopwords_split_ok = check_data_split_sizes(
        data_dir / 'train.csv',
        data_dir / 'val.csv',
        data_dir / 'test.csv'
    )
    
    without_stopwords_split_ok = check_data_split_sizes(
        data_dir / 'train_ns.csv',
        data_dir / 'val_ns.csv',
        data_dir / 'test_ns.csv'
    )
    
    result = with_stopwords_split_ok and without_stopwords_split_ok
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"\nData split ratios are correct: {status}")
    
    return result

def run_stopwords_tests(data_dir):
    """Test if stopwords are correctly removed."""
    print("\n----- Testing stopwords removal -----")
    
    stopwords_ok = check_stopwords_usage(data_dir)
    
    status = "✅ PASS" if stopwords_ok else "❌ FAIL"
    print(f"\nStopwords removal is correct: {status}")
    
    return stopwords_ok

def run_all_tests(data_dir, script_path='main.py'):
    """Run all tests."""
    print("\n========== Testing Assignment 1 Output Files ==========\n")
    
    # Check if input data files exist
    pos_path = data_dir / 'pos.txt'
    neg_path = data_dir / 'neg.txt'
    stopwords_path = data_dir / 'stopwords.txt'
    
    input_files_exist = True
    
    if not os.path.isfile(pos_path):
        print(f"❌ Input file {pos_path} not found. Please download it first.")
        input_files_exist = False
    
    if not os.path.isfile(neg_path):
        print(f"❌ Input file {neg_path} not found. Please download it first.")
        input_files_exist = False
    
    if not os.path.isfile(stopwords_path):
        print(f"❌ Input file {stopwords_path} not found. Please download it first.")
        input_files_exist = False
    
    if not input_files_exist:
        print("\nPlease make sure the required input files exist in the data directory.")
        print("You can download them using the commands in the README.md file.")
        return 1
    
    # Run main.py to generate output files
    print("\n----- Running main.py to generate output files -----")
    try:
        import subprocess
        result = subprocess.run(['python', script_path, str(data_dir)], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"❌ Failed to run {script_path}. Error:")
            print(result.stderr)
            return 1
        
        print(f"✅ Successfully ran {script_path}")
        print(result.stdout)
    except Exception as e:
        print(f"❌ Error running {script_path}: {e}")
        return 1
    
    # Run all tests
    files_result = run_file_existence_tests(data_dir)
    format_result = run_format_tests(data_dir)
    split_result = run_split_tests(data_dir)
    stopwords_result = run_stopwords_tests(data_dir)
    import_result = run_import_check(script_path)
    label_result = run_label_tests(data_dir)
    
    # Summary
    print("\n========== Test Summary ==========")
    print(f"1. All files exist and not empty: {'✅ PASS' if files_result else '❌ FAIL'}")
    print(f"2. All files follow CSV format: {'✅ PASS' if format_result else '❌ FAIL'}")
    print(f"3. Data split ratios are correct: {'✅ PASS' if split_result else '❌ FAIL'}")
    print(f"4. Stopwords removal is correct: {'✅ PASS' if stopwords_result else '❌ FAIL'}")
    print(f"5. No unauthorized libraries: {'✅ PASS' if import_result else '❌ FAIL'}")
    print(f"6. Label files are correct: {'✅ PASS' if label_result else '❌ FAIL'}")
    
    all_tests_passed = (files_result and format_result and split_result and 
                        stopwords_result and import_result and label_result)
    
    print("\n========== Final Result ==========")
    if all_tests_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED. Please fix the issues before submitting.")
    
    return 0 if all_tests_passed else 1

def main():
    parser = argparse.ArgumentParser(description='Test script for MSCI 641 Assignment 1')
    parser.add_argument('data_dir', nargs='?', default='./data', help='Directory containing the data files')
    parser.add_argument('--check-files', action='store_true', help='Check if all required files exist')
    parser.add_argument('--check-formats', action='store_true', help='Check if files follow the required CSV format')
    parser.add_argument('--check-splits', action='store_true', help='Check if data split follows 80/10/10 ratio')
    parser.add_argument('--check-stopwords', action='store_true', help='Check if stopwords are correctly removed')
    parser.add_argument('--check-imports', action='store_true', help='Check for unauthorized libraries')
    parser.add_argument('--check-labels', action='store_true', help='Check if label files are correct')
    parser.add_argument('--script-path', type=str, default='main.py', help='Path to the main script to check for imports')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    script_path = args.script_path
    
    # If no specific tests are requested, run all tests
    if not (args.check_files or args.check_formats or args.check_splits or 
            args.check_stopwords or args.check_imports or args.check_labels) or args.all:
        return run_all_tests(data_dir, script_path)
    
    # Run only requested tests
    results = []
    
    if args.check_files:
        results.append(run_file_existence_tests(data_dir))
    
    if args.check_formats:
        results.append(run_format_tests(data_dir))
    
    if args.check_splits:
        results.append(run_split_tests(data_dir))
    
    if args.check_stopwords:
        results.append(run_stopwords_tests(data_dir))
    
    if args.check_imports:
        results.append(run_import_check(script_path))
    
    if args.check_labels:
        results.append(run_label_tests(data_dir))
    
    # Return success if all requested tests pass
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())