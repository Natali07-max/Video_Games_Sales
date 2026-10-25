# 🎮 Video Game Sales Prediction

A Machine Learning web application that predicts global video game sales using game attributes such as platform, release year, genre, and publisher.

## 📌 Project Overview

**Video Game Sales Prediction** is a machine learning project developed using Python and Flask. It uses historical video game sales data to estimate the expected global sales of a game based on selected attributes.

The application integrates a trained **Random Forest Regressor** model with a user-friendly web interface. Users can select game details, generate a sales prediction, and view the model performance dashboard.

## ✨ Features

- 🎯 Machine Learning-based global sales prediction
- 🎮 Interactive input form with dropdown selections
- 📊 Model performance dashboard
- ⚡ Automated predictions using a trained model
- 🌐 Web interface built with Flask, HTML, and CSS
- 🧩 Reusable model and label encoders loaded using Joblib

## 🧠 Machine Learning Workflow

1. Load and inspect the historical video game sales dataset.
2. Handle missing values and prepare the data.
3. Encode categorical features using Label Encoding.
4. Split the dataset into training and testing sets.
5. Train a Random Forest Regressor.
6. Save the trained model and label encoders using Joblib.
7. Load the saved model into the Flask application.
8. Collect game details through the web interface.
9. Encode the selected input values.
10. Predict and display the estimated global sales.

## 📥 Input Features

The model uses the following four input features:

| Feature | Description |
|---|---|
| Platform | Gaming platform on which the game was released |
| Year | Game release year |
| Genre | Category or type of game |
| Publisher | Company that published the game |

## 📤 Prediction Output

**Global Sales** is the target variable predicted by the model.

The prediction is based on the relationships learned from historical game sales data. The displayed value uses the same unit as the target variable used during model training.

## 📊 Model Performance Dashboard

The application includes a dashboard for viewing the model evaluation metrics stored in `model_metrics.json`.

The dashboard can be accessed at:

`http://127.0.0.1:5000/dashboard`

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Web application framework |
| Scikit-learn | Machine learning and model training |
| Pandas | Dataset preparation and analysis |
| Joblib | Saving and loading the trained model |
| HTML | Web page structure |
| CSS | User interface styling |
| JSON | Storing model evaluation metrics |

## 📁 Project Structure

```text
Video_Games_Sales/
├── app.py
├── game_sales_model.pkl
├── label_encoders.pkl
├── model_metrics.json
├── templates/
│   ├── index.html
│   ├── result.html
│   └── dashboard.html
└── README.md
```

*Note: The structure above reflects the expected core files. Keep only the files and folders that actually exist in your repository.*

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip
- Visual Studio Code or another Python editor

### Installation

1. Clone or download this repository.
2. Open the project folder in your terminal.
3. Install the required dependencies:

```bash
pip install flask joblib scikit-learn pandas numpy
```

4. Ensure that the trained model, label encoders, and metrics file are available in the project folder.
5. Start the Flask application:

```bash
python app.py
```

6. Open the application in your browser:

`http://127.0.0.1:5000/`

## 🔮 Future Enhancements

- Improve prediction accuracy through model comparison and hyperparameter tuning.
- Add exploratory data analysis and interactive sales visualizations.
- Compare predictions with actual sales where data is available.
- Deploy the application to a cloud hosting platform.
- Improve model evaluation and input validation.

## 👩‍💻 Author

**Natali Sonawane**

B.E. Information Technology Student

GitHub: [Natali07-max](https://github.com/Natali07-max)

---

*This project demonstrates the integration of machine learning with a web application for predictive analytics.*
