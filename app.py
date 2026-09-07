from flask import Flask, render_template, request
import joblib
import os

from src.preprocess import clean_text


app = Flask(__name__)


# Load trained ML model
MODEL_PATH = os.path.join(
    "models",
    "sentiment_model.pkl"
)

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    cleaned_text = clean_text(text)

    prediction = model.predict([cleaned_text])[0]

    return render_template(
        "index.html",
        text=text,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)