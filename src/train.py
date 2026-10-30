import os
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

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

# train.py is inside src folder
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Main project folder is one level above src
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sentiment_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# Import preprocessing function from same src folder
from preprocess import clean_text

print("Project directory:", BASE_DIR)
print("Dataset path:", DATA_PATH)
print("Model directory:", MODEL_DIR)

# Check dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found. Please check: {DATA_PATH}"
    )

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

# Check required columns
if not {"text", "sentiment"}.issubset(df.columns):
    raise ValueError(
        "CSV must contain 'text' and 'sentiment' columns."
    )

print("\nDataset preview:")
print(df.head())

print("\nOriginal dataset shape:")
print(df.shape)

# --------------------------------------------------
# 3. CLEAN DATASET
# --------------------------------------------------

df = df.dropna(subset=["text", "sentiment"])
df = df.drop_duplicates()

df["text"] = df["text"].astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

print("\nDataset after cleaning:")
print(df.shape)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

if df["sentiment"].nunique() < 2:
    raise ValueError("Dataset must contain at least two sentiment classes.")

if df["sentiment"].value_counts().min() < 2:
    raise ValueError(
        "Each sentiment class must have at least two records for stratified splitting."
    )

# --------------------------------------------------
# 4. TEXT PREPROCESSING
# --------------------------------------------------

print("\nCleaning text...")

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["sentiment"]

# --------------------------------------------------
# 5. TRAIN TEST SPLIT
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
# 6. CREATE MODELS
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
            LogisticRegression(max_iter=1000)
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
# 7. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = {}

best_model = None
best_model_name = None
best_f1 = -1

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# --------------------------------------------------
# 8. SAVE MODEL COMPARISON RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results).T

print("\nMODEL COMPARISON")
print("=" * 60)
print(results_df)

results_path = os.path.join(
    MODEL_DIR,
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index_label="Model"
)

print("\nResults saved at:", results_path)

# --------------------------------------------------
# 9. SAVE BEST MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)

joblib.dump(best_model, MODEL_PATH)

print("\nBest model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))
print("Model saved at:", MODEL_PATH)

# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

class_labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=class_labels
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

confusion_path = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)

plt.savefig(confusion_path, dpi=300)

print("\nConfusion matrix saved at:", confusion_path)

plt.show()

# --------------------------------------------------
# 11. TRAINING COMPLETED
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Dataset records:", len(df))
print("Best model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))
print("Results file:", results_path)
print("Model file:", MODEL_PATH)
print("Confusion matrix:", confusion_path)