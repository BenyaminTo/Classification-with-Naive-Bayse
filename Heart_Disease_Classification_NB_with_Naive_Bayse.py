# %% [markdown]
# # Heart Disease Classification with Naive Bayes
#
# This project uses Multinomial Naive Bayes to classify heart disease data.
# The notebook includes data splitting, model training, classification report,
# ROC curve calculation, ROC curve plotting, and K-fold cross-validation.

# %% [markdown]
# ## 1. Import Libraries
#
# Import the required libraries for data processing, model training,
# evaluation metrics, and visualization.

# %%
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# %% [markdown]
# ## 2. Load the Heart Disease Dataset
#
# Load the dataset and inspect its structure, columns, and first rows.

# %%
# Define the dataset path
data_path = "/Heart-Disease-UCI.csv"

# Load the dataset
data = pd.read_csv(data_path)

# Show dataset information
data.info()

# Show the first rows of the dataset
print(data.head())


# %% [markdown]
# ## 3. Define Features and Split Data
#
# Use all columns except the last column as features.
# The last column is considered the target label.
# Then split the data into training and test sets.

# %%
# Define features and labels
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

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


# %% [markdown]
# ## 4. Train and Evaluate the Naive Bayes Model
#
# Train a Multinomial Naive Bayes classifier on the training data,
# then predict labels and probabilities on the test data.

# %%
# Create and train the Multinomial Naive Bayes classifier
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

# Predict class probabilities for the test set
prediction_prob = clf.predict_proba(X_test)

# Show predicted probabilities for the first 10 samples
print("Predicted probabilities for first 10 samples:")
print(prediction_prob[0:10])

# Predict class labels for the test set
prediction = clf.predict(X_test)

# Show predicted labels for the first 10 samples
print("Predicted labels for first 10 samples:")
print(prediction[:10])

# Calculate accuracy on the test set
accuracy = clf.score(X_test, Y_test)
print(f"The accuracy is: {accuracy * 100:.1f}%")


# %% [markdown]
# ## 5. Classification Report
#
# Print precision, recall, F1-score, and support for each class.

# %%
# Create the classification report
report = classification_report(Y_test, prediction)

# Print the classification report
print("\nClassification Report:")
print(report)


# %% [markdown]
# ## 6. Manual ROC Curve Calculation
#
# Calculate the True Positive Rate and False Positive Rate manually
# using different probability thresholds.
#
# Note: This section assumes that the target is binary and encoded as 0 and 1,
# where 1 is the positive class.

# %%
# Probability of the positive class
pos_prob = prediction_prob[:, 1]

# Thresholds for ROC curve calculation
thresholds = np.arange(0.0, 1.1, 0.05)

# Initialize True Positive and False Positive counters
true_pos = [0] * len(thresholds)
false_pos = [0] * len(thresholds)

# Count TP and FP for each threshold
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        # If the predicted probability is greater than or equal to the threshold,
        # the sample is considered positive.
        if pred >= threshold:
            if y == 1:
                # True Positive: actual label is 1 and predicted label is also 1
                true_pos[i] += 1
            else:
                # False Positive: actual label is 0 but predicted label is 1
                false_pos[i] += 1
        else:
            # Since thresholds are increasing,
            # larger thresholds will also be greater than pred.
            break

# Number of positive and negative samples in the test set
n_pos_test = int((Y_test == 1).sum())
n_neg_test = int((Y_test == 0).sum())

# Compute True Positive Rate and False Positive Rate
# Add a guard to avoid division by zero
true_pos_rate = np.array([
    tp / n_pos_test if n_pos_test > 0 else 0.0
    for tp in true_pos
])

false_pos_rate = np.array([
    fp / n_neg_test if n_neg_test > 0 else 0.0
    for fp in false_pos
])

# The thresholds are increasing, so ROC points are generated from (1,1) toward (0,0).
# Reverse them to plot the ROC curve in the usual direction.
true_pos_rate = true_pos_rate[::-1]
false_pos_rate = false_pos_rate[::-1]


# %% [markdown]
# ## 7. Plot ROC Curve
#
# Plot the ROC curve using the computed True Positive Rate
# and False Positive Rate values.

# %%
# Create the ROC curve plot
plt.figure()
lw = 2

# Plot the ROC curve
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw)

# Plot the diagonal chance line
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Set plot limits
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Set axis labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

# Show legend
plt.legend(loc="lower right")

# Display the plot
plt.show()

# Print ROC-AUC score
print("ROC-AUC Score:", roc_auc_score(Y_test, pos_prob))


# %% [markdown]
# ## 8. K-Fold Cross-Validation
#
# Perform 5-fold stratified cross-validation to evaluate
# model stability using accuracy and ROC-AUC.

# %%
# Number of folds
k = 5

# Create stratified K-fold cross-validation
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Lists to store results of each fold
accuracies = []
auc_scores = []

print("--- Start K-Fold Validation ---")

# Loop through each fold
for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
    print(f"\n--- Fold {fold + 1} ---")

    # Split data into training and validation sets for this fold
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    Y_train_fold, Y_val_fold = Y.iloc[train_index], Y.iloc[val_index]

    # Create and train a new model for this fold
    clf = MultinomialNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train_fold, Y_train_fold)

    # Predict labels and probabilities
    Y_pred_fold = clf.predict(X_val_fold)
    Y_prob_fold = clf.predict_proba(X_val_fold)[:, 1]

    # Calculate evaluation metrics
    acc = accuracy_score(Y_val_fold, Y_pred_fold)
    auc = roc_auc_score(Y_val_fold, Y_prob_fold)

    # Store results
    accuracies.append(acc)
    auc_scores.append(auc)

    # Print results for this fold
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"AUC: {auc:.4f}")

# Print final K-fold results
print("\n" + "=" * 30)
print("--- K-Fold Final Results ---")
print(f"Mean Accuracy: {np.mean(accuracies) * 100:.2f}% (+/- {np.std(accuracies) * 100:.2f}%)")
print(f"Mean AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")