#!/usr/bin/env python
# coding: utf-8

# Import required libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedGroupKFold


# Load the Movie Dataset

# In[4]:


# Load the MovieLens dataset and inspect its first rows, columns,
# data types, and missing values.

# Define the dataset path
data_path = "C:/Project/Data/ml-latest-small/movies.csv"

# Load the movie dataset
data = pd.read_csv(data_path)

# Show the first rows
print(data.head())

# Show dataset information
data.info()

# Show the number of missing values in each column
print(data.isnull().sum())


# Clean the Text Data

# In[5]:


# Remove the movieId column because it is not needed for text classification
df = data.drop(columns=["movieId"])

# Keep only rows where title and genres are available
df = df.dropna(subset=["title", "genres"])

# Define a function to clean movie titles
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\(\d{4}\)", "", text)       # Remove the year in parentheses
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # Keep only letters and spaces
    text = re.sub(r"\s+", " ", text)            # Remove extra whitespace
    return text.strip()

# Create a cleaned title column
df["title_clean"] = df["title"].apply(clean_text)


# Split the Data

# In[6]:


# Define features and labels
X = df["title_clean"]
Y = df["genres"]

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Show the number of samples in training and test sets
print("Training samples:", len(Y_train))
print("Test samples:", len(Y_test))


# Vectorize Text and Encode Multi-label Genres

# In[7]:


# Convert cleaned titles into numerical feature vectors
vectorizer = CountVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Split genre strings by the pipe separator
# Example: "Action|Adventure|Sci-Fi" -> ["Action", "Adventure", "Sci-Fi"]
Y_train_split = Y_train.apply(lambda x: x.split('|'))
Y_test_split = Y_test.apply(lambda x: x.split('|'))

# Convert genre lists into binary multi-label vectors
mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(Y_train_split)
Y_test_bin = mlb.transform(Y_test_split)


# Train the Multi-label Naive Bayes Model

# In[8]:


# Train a One-vs-Rest Multinomial Naive Bayes classifier
clf = OneVsRestClassifier(MultinomialNB(alpha=1.0, fit_prior=True))

# Fit the model on training data
clf.fit(X_train_vec, Y_train_bin)

# Predict genre probabilities for the test set
prediction_prob = clf.predict_proba(X_test_vec)

# Show predicted probabilities for the first five samples
print("Probabilities for first 5 samples:")
print(prediction_prob[0:5])


# Predict Genres and Calculate Exact Match Accuracy

# In[9]:


# Predict binary genre labels for the test set
prediction = clf.predict(X_test_vec)

# Convert binary predictions back to readable genre names
predicted_genres = mlb.inverse_transform(prediction)

# Show predicted genres for the first five samples
print("\nPredicted genres for first 5 samples:")
print(predicted_genres[:5])

# Calculate exact match accuracy for multi-label prediction
exact_match_accuracy = np.mean(prediction == Y_test_bin)
print(f"\nExact Match Accuracy: {exact_match_accuracy * 100:.1f}%")


# Classification Report

# In[10]:


# Print a detailed classification report for each genre
print("\nClassification Report:")
print(
    classification_report(
        Y_test_bin,
        prediction,
        target_names=mlb.classes_,
        zero_division=0
    )
)


# Compute ROC Metrics

# In[11]:


# Ensure that prediction probabilities are available
if prediction_prob is None:
    raise ValueError(
        "Critical error: prediction_prob is None. "
        "Make sure clf.predict_proba(X_test_vec) has been executed."
    )

# Convert dense or sparse data to a one-dimensional NumPy array
def get_1d_array(data):
    if sp.issparse(data):
        return data.toarray().ravel()
    return np.asarray(data).ravel()

# Get the number of genre classes
n_classes = Y_test_bin.shape[1]

# Create dictionaries to store ROC metrics
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and AUC for each genre separately
for i in range(n_classes):
    y_true_i = get_1d_array(Y_test_bin[:, i])
    y_score_i = get_1d_array(prediction_prob[:, i])

    fpr[i], tpr[i], _ = roc_curve(y_true_i, y_score_i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC across all genres
y_true_micro = get_1d_array(Y_test_bin)
y_score_micro = get_1d_array(prediction_prob)

fpr["micro"], tpr["micro"], _ = roc_curve(y_true_micro, y_score_micro)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Plot ROC Curves

# In[12]:


# Create the ROC curve plot
plt.figure(figsize=(10, 8))
lw = 2

# Plot the micro-average ROC curve
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
    color='deeppink',
    linestyle=':',
    linewidth=4
)

# Plot ROC curves for the first five genres
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
classes_to_plot = range(min(5, n_classes))

for i, color in zip(classes_to_plot, colors):
    genre_name = mlb.classes_[i]
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label=f'{genre_name} (AUC = {roc_auc[i]:.2f})'
    )

# Plot the diagonal chance line
plt.plot([0, 1], [0, 1], 'k--', lw=lw)

# Set plot limits
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Set axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Multi-label)')

# Show legend and grid
plt.legend(loc="lower right", fontsize='small')
plt.grid(alpha=0.3)

# Display the plot
plt.show()

# Print micro-average ROC-AUC score
print(f"Micro-average ROC-AUC Score: {roc_auc['micro']:.4f}")

