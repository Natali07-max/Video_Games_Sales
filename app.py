import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
model = joblib.load("game_sales_model.pkl")

# Load label encoders
encoders = joblib.load("label_encoders.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    platform = request.form["platform"]
    year = int(request.form["year"])
    genre = request.form["genre"]
    publisher = request.form["publisher"]

    # Convert text into numbers
    platform = encoders["Platform"].transform([platform])[0]
    genre = encoders["Genre"].transform([genre])[0]
    publisher = encoders["Publisher"].transform([publisher])[0]

    # Predict
    prediction = model.predict([[platform, year, genre, publisher]])

    return render_template(
        "result.html",
        prediction=round(prediction[0], 2)
    )


if __name__ == "__main__":
    app.run(debug=True)