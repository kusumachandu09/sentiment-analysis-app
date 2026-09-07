import os
import sys

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Allow importing preprocess.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "sentiment_data.csv"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# --------------------------------------------------
# 2. Remove missing values
# --------------------------------------------------

df = df.dropna(subset=["text", "sentiment"])

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset after cleaning:")
print(df.shape)


# --------------------------------------------------
# 3. Clean text
# --------------------------------------------------

print("\nCleaning text...")

df["clean_text"] = df["text"].apply(clean_text)


# --------------------------------------------------
# 4. Input and output
# --------------------------------------------------

X = df["clean_text"]
y = df["sentiment"]


# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. Create ML models
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]),

    "Naive Bayes": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2)
            )
        ),

        (
            "classifier",
            MultinomialNB()
        )
    ]),

    "Linear SVM": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LinearSVC()
        )
    ])
}


# --------------------------------------------------
# 7. Train and evaluate
# --------------------------------------------------

results = {}

best_model = None
best_model_name = None
best_f1 = 0


for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Select best model using F1 score
    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_model_name = name


# --------------------------------------------------
# 8. Print comparison
# --------------------------------------------------

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results).T

print(results_df)


# --------------------------------------------------
# 9. Save best model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)

joblib.dump(best_model, MODEL_PATH)

print("\nBest model:", best_model_name)

print("Best F1 Score:", round(best_f1, 4))

print("\nModel saved at:")
print(MODEL_PATH)


# --------------------------------------------------
# 10. Confusion matrix
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=sorted(y.unique()),
    yticklabels=sorted(y.unique())
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

confusion_path = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)

plt.savefig(confusion_path)

plt.show()


# --------------------------------------------------
# 11. Save model comparison
# --------------------------------------------------

results_path = os.path.join(
    MODEL_DIR,
    "model_results.csv"
)

results_df.to_csv(results_path)

print("\nResults saved at:")
print(results_path)

print("\nTraining completed successfully!")