# 🎮 Video Game Sales Prediction
---

## 📌 Project Overview

The **Video Game Sales Prediction** project is a Machine Learning-based web application developed using **Python** and **Flask**. It analyzes historical video game sales data and predicts the expected **Global Sales** based on various game attributes such as platform, release year, genre, publisher.

The application utilizes a trained **Random Forest Regressor** model to provide accurate predictions. It also generates graphical visualizations that help users understand  sales distribution and compare predicted global sales.

---

## ✨ Features

- 🎯 AI-Based Global Sales Prediction
- 📊 Dynamic Graph Generation
- 🎮 User-Friendly Web Interface
- ⚡ Fast and Accurate Predictions
- 📱 Responsive Design

---

## 🧠 Machine Learning Workflow

1. Load the Video Game Sales dataset.
2. Handle missing values.
3. Encode categorical features using Label Encoding.
4. Split the dataset into training and testing sets.
5. Train the Random Forest Regression model.
6. Save the trained model using Joblib.
7. Load the trained model in the Flask application.
8. Accept user input through the web interface.
9. Predict the Global Sales.
10. Display the prediction along with graphical visualization.

---

## 📊 Input Parameters

- Platform
- Year
- Genre
---

## 📈 Output

- Predicted Global Sales
- Regional Sales Graph
- Sales Visualization

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Flask | Web Framework |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| Matplotlib | Graph Visualization |
| HTML | Frontend Structure |
| CSS | Styling |
| Joblib | Model Serialization |


## 📷 Application Workflow

1. Launch the Flask application.
2. Enter the required game details.
3. Click the **Predict** button.
4. The trained Machine Learning model predicts the expected **Global Sales**.
5. The application displays the predicted value.
6. A graph is generated to visualize the regional sales and prediction.


The dataset includes:

- Platform
- Year
- Genre
- Publisher
- NA Sales
- EU Sales
- JP Sales
- Other Sales
- Global Sales But I have only included 4 and "Global Sales" as Output






