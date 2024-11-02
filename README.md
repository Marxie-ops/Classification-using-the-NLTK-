# Natural-Language-Processing
 ##  Spam Detection Using Naive Bayes Algorithm

This project implements a classification model using the Naive Bayes algorithm to predict whether a given SMS message is spam or not.

**Objective:**

Build a classifier to distinguish spam messages from legitimate (ham) messages.
Utilize text data preprocessing and the Naive Bayes algorithm for accurate predictions.
Dataset:

The "SMS Spam Collection" dataset is used, publicly available on the UCI Machine Learning Repository.
It contains labeled SMS messages categorized as "spam" or "ham".
Project Workflow:

**Project Setup**
Install necessary libraries (pandas, numpy, scikit-learn, nltk).
Load and explore the dataset.
## **Data Preprocessing**
**Text cleaning:** Lowercasing, removing stopwords, punctuation, and applying stemming/lemmatization.
Vectorization: Convert text data into numerical features using techniques like Bag of Words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF).
Split the dataset into training and testing sets.

## **Model Building (Naive Bayes)**
Train a Multinomial Naive Bayes classifier on the training data.
Test the model on unseen data (test set) and evaluate its performance.

## **Model Evaluation**
Measure accuracy, precision, recall, and F1-score to assess the model's effectiveness.
Plot a confusion matrix to visualize the classification results.

## **Model Interpretation**
Identify the most influential features contributing to spam classification.
Analyze common words in spam and ham messages using techniques like word clouds or bar plots.
Conclusion
Summarize key findings and potential improvements for further development.

# Report: Naive Bayes Classification Results

1. **Model Overview**

This model employs a Naive Bayes classifier to classify SMS messages as "spam" or "ham". Its performance is evaluated using various metrics (accuracy, precision, recall, F1-score) and a confusion matrix on a test set of 1,115 samples.

2. **Overall Accuracy**

The model achieves a remarkable accuracy of 98.12%, indicating successful classification of 98.12% of samples in the test set.

3. **Detailed Metrics**

**Precision:** Measures the proportion of true positive predictions (correctly classified as "spam" or "ham").
False (ham): 0.98 (98% of messages predicted as "ham" were actually ham).
True (spam): 1.00 (perfect precision, all messages predicted as "spam" were indeed spam).
**Recall:** Represents the proportion of actual positives (spam or ham) correctly identified by the model.
False (ham): 1.00 (all ham messages were correctly identified).
True (spam): 0.87 (87% of actual spam messages were correctly identified).
**F1-Score:** Harmonic mean of precision and recall, balancing both metrics.
False (ham): 0.99 (indicating near-perfect classification).
True (spam): 0.93 (reflects good spam detection, with room for improvement).
4. **Confusion Matrix**

Predicted	True Negatives (TN)	False Positives (FP)
False (ham)	955	0
True (spam)	21	139

**Export to Sheets**
True Negatives (TN): 955 ham messages correctly classified.
False Positives (FP): 0 ham messages incorrectly classified as spam (no false positives).
False Negatives (FN): 21 spam messages incorrectly classified as ham.
True Positives (TP): 139 spam messages correctly classified.
5. **Insights and Recommendations**

**High Precision (Spam):** The model excels at identifying spam messages when they are predicted.
**Good Overall Performance:** The model performs very well, especially for ham messages, demonstrating both high recall and precision.
Room for Improvement in Spam Detection: While spam precision is perfect, the recall for spam is lower (0.87), indicating some missed spam messages. Techniques like adjusting the decision threshold or refining preprocessing for spam-related features could improve this.
6. **Conclusion**

The Naive Bayes classifier achieves a remarkable accuracy of 98.12% with strong precision and recall, particularly for ham messages. While spam detection is effective, further adjustments could help capture more spam messages while maintaining precision. This model is well-suited for














