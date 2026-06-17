# MSE 641 Assignment 4 Report

**Student Name:** Li Xiaohui
**Student ID:** 21221482
**Date:** 2026-06-17

*Maximum 10 sentences total for all analysis sections below.*

## Final Results Summary

| Activation Function | Best L2 Regularization | Best Dropout Rate | Validation Accuracy | Test Accuracy |
|---------------------|------------------------|-------------------|---------------------|---------------|
| ReLU                | 0.001                  | 0.3               | 79.21%              | 78.58%        |
| Sigmoid             | 0.001                  | 0.3               | 77.99%              | 77.60%        |
| Tanh                | 0.001                  | 0.3               | 78.51%              | 77.93%        |

## Analysis (Maximum 10 sentences total)

### Effect of Activation Functions (2-3 sentences)
*Which activation function performed best and why do you think that is?*

ReLU achieved the best performance (78.58% test accuracy), followed by Tanh (77.93%) and Sigmoid (77.60%). ReLU's superior performance can be attributed to its ability to avoid the vanishing gradient problem and its computational efficiency, making it well-suited for training deep networks on sentiment classification tasks.

### Effect of L2 Regularization (2-3 sentences)
*How did L2 regularization affect your model performance? Did it help prevent overfitting?*

Lower L2 regularization (0.001) consistently outperformed higher values (0.01) across all activation functions. Notably, Sigmoid with L2=0.01 showed a dramatic drop to 51.68% accuracy, indicating that excessive regularization can severely harm models with bounded activation functions by over-constraining the weight updates.

### Effect of Dropout (2-3 sentences)
*How did dropout affect your results? What dropout rates worked best?*

A dropout rate of 0.3 yielded better results than 0.5 for ReLU (79.21% vs 78.88%). This suggests that moderate dropout provides sufficient regularization for this sentiment analysis task, while higher dropout rates may discard too much information and hinder model learning.

### Best Configuration and Key Insight (2-3 sentences)
*What was your best overall model and why do you think this combination worked well? What was the most important thing you learned from this assignment?*

**Best Model:** ReLU with L2=0.001, Dropout=0.3, Test Accuracy=78.58%

This combination worked well because ReLU's non-saturating nature combined with moderate regularization (low L2 and 0.3 dropout) provided an optimal balance between model capacity and generalization. The key insight from this assignment is that hyperparameter interactions matter significantly - the same L2 value that works well for ReLU can catastrophically harm Sigmoid, highlighting the importance of systematic hyperparameter tuning.
