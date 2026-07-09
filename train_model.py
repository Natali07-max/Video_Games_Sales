import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Dataset/vgsales.csv")
# If your file is inside Dataset folder, use:
# df = pd.read_csv("Dataset/vgsales.csv")

# -----------------------------
# Handle Missing Values
# -----------------------------
df["Year"] = df["Year"].fillna(df["Year"].median())
df["Publisher"] = df["Publisher"].fillna(df["Publisher"].mode()[0])

import matplotlib.pyplot as plt

platform_sales = df.groupby("Platform")["Global_Sales"].sum()
top10 = platform_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(top10.index, top10.values)

plt.title("Top 10 Platforms by Global Sales")
plt.xlabel("Platform")
plt.ylabel("Global Sales (Millions)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("static/platform_sales.png")

plt.close()

print("Graph Saved Successfully")

# -----------------------------
# Create Label Encoders
# -----------------------------
platform_encoder = LabelEncoder()
genre_encoder = LabelEncoder()
publisher_encoder = LabelEncoder()

df["Platform"] = platform_encoder.fit_transform(df["Platform"])
df["Genre"] = genre_encoder.fit_transform(df["Genre"])
df["Publisher"] = publisher_encoder.fit_transform(df["Publisher"])

# -----------------------------
# Print Available Values (for debugging)
# -----------------------------
print("Platforms:", platform_encoder.classes_)
print("Genres:", genre_encoder.classes_)
print("Publishers:", publisher_encoder.classes_[:10])   # First 10 publishers

# -----------------------------
# Select Features
# -----------------------------
X = df[["Platform", "Year", "Genre", "Publisher"]]
y = df["Global_Sales"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("------------------")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "game_sales_model.pkl")

# -----------------------------
# Save Encoders
# -----------------------------
encoders = {
    "Platform": platform_encoder,
    "Genre": genre_encoder,
    "Publisher": publisher_encoder
}

joblib.dump(encoders, "label_encoders.pkl")

print("\nModel and Encoders saved successfully!")
