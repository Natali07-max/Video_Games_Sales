
import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# -------------------- LOAD MODEL --------------------

MODEL_PATH = BASE_DIR / "game_sales_model.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not ENCODERS_PATH.exists():
    raise FileNotFoundError(f"Encoder file not found: {ENCODERS_PATH}")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

REQUIRED_ENCODERS = ["Platform", "Genre", "Publisher"]

for column in REQUIRED_ENCODERS:
    if column not in encoders:
        raise KeyError(f"Missing encoder for {column} in label_encoders.pkl")


# -------------------- HELPER FUNCTIONS --------------------

def get_form_options():
    """Return valid categories from the saved training encoders."""
    return {
        "platforms": sorted(
            encoders["Platform"].classes_.tolist(),
            key=lambda value: str(value).casefold()
        ),
        "genres": sorted(
            encoders["Genre"].classes_.tolist(),
            key=lambda value: str(value).casefold()
        ),
        "publishers": sorted(
            encoders["Publisher"].classes_.tolist(),
            key=lambda value: str(value).casefold()
        ),
    }


def encode_category(column, value):
    """
    Encode a form value using the original training encoder.
    Matching ignores surrounding whitespace and letter case.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Please select a valid {column}.")

    value = value.strip()
    encoder = encoders[column]

    for label in encoder.classes_:
        if str(label).strip().casefold() == value.casefold():
            return int(encoder.transform([label])[0])

    raise ValueError(
        f"'{value}' is not supported for {column}. "
        "Please select an available option from the dropdown."
    )


def render_form(error=None, status=200):
    """Render the prediction form with its valid dropdown options."""
    return (
        render_template(
            "index.html",
            **get_form_options(),
            error=error
        ),
        status
    )


# -------------------- HOME PAGE --------------------

@app.route("/")
def home():
    return render_form()


# -------------------- PREDICTION --------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        platform = request.form.get("platform", "")
        genre = request.form.get("genre", "")
        publisher = request.form.get("publisher", "")
        year_text = request.form.get("year", "").strip()

        # Validate release year
        if not year_text:
            raise ValueError("Please enter a release year.")

        try:
            year = int(year_text)
        except ValueError:
            raise ValueError("Release year must be a whole number.")

        if year < 1970 or year > 2035:
            raise ValueError(
                "Please enter a release year between 1970 and 2035."
            )

        # Use the same encoders used during model training
        platform_encoded = encode_category("Platform", platform)
        genre_encoded = encode_category("Genre", genre)
        publisher_encoded = encode_category("Publisher", publisher)

        # Feature order must match the model's training order:
        # Platform, Year, Genre, Publisher
        features = np.array(
            [[platform_encoded, year, genre_encoded, publisher_encoded]],
            dtype=float
        )

        # Generate prediction
        prediction = float(model.predict(features)[0])

        if not np.isfinite(prediction):
            raise ValueError(
                "The model returned an invalid prediction. "
                "Please check the trained model."
            )

        # Video game sales are normally measured in millions.
        # This assumes the target used during training was Global_Sales.
        return render_template(
            "result.html",
            prediction=round(max(0.0, prediction), 2),
            platform=platform,
            year=year,
            genre=genre,
            publisher=publisher
        )

    except (ValueError, KeyError) as error:
        return render_form(str(error), 400)

    except Exception:
        app.logger.exception("Prediction failed")
        return render_form(
            "Prediction failed because of a model or configuration error. "
            "Check the VS Code terminal for details.",
            500
        )


# -------------------- DASHBOARD --------------------

@app.route("/dashboard")
def dashboard():
    metrics_path = BASE_DIR / "model_metrics.json"

    if not metrics_path.exists():
        return (
            "Dashboard metrics file not found. "
            "Please generate model_metrics.json during training.",
            404
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        return render_template("dashboard.html", metrics=metrics)

    except (json.JSONDecodeError, OSError):
        app.logger.exception("Could not load dashboard metrics")
        return "Could not load dashboard metrics.", 500


# -------------------- RUN APPLICATION --------------------

if __name__ == "__main__":
    app.run(debug=True)
