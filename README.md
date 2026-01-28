# 🏠 House Price Prediction using EDA, ANN & Machine Learning

## 📌 Project Overview
This project aims to predict house prices using the Housing dataset. It covers the complete data science workflow including Exploratory Data Analysis (EDA), feature preprocessing, and regression modeling using Artificial Neural Networks (ANN) and ensemble-based machine learning models.

---

## 📂 Dataset
- **File:** `Housing.csv`
- **Total Records:** 545
- **Features:** Area, bedrooms, bathrooms, stories, parking, and multiple categorical attributes
- **Target Variable:** `price`

---

## 🔍 Exploratory Data Analysis (EDA)
EDA is performed to understand the structure and behavior of the dataset using multiple visualization techniques:
- Distribution analysis using histograms
- Outlier detection using boxplots
- Categorical analysis using bar, horizontal bar, and pie charts
- Feature relationships using scatter and line plots
- Correlation analysis using heatmaps

📘 **Notebook:** `House_Price_Prediction_EDA.ipynb`

---

## 🧠 Feature Engineering
- Removal of unnecessary column (`Unnamed: 12`)
- Encoding of categorical variables using one-hot encoding
- Feature scaling applied for ANN models
- Train-test split for model evaluation

---

## 🤖 Models Implemented

### 🔹 Artificial Neural Network (ANN)
- Feedforward neural network for regression
- ReLU activation with He initialization
- Linear activation in the output layer
- Optimized using Adam optimizer and MSE loss

📘 **Notebook:** `House_Price_Prediction_ANN.ipynb`

### 🔹 Machine Learning Models
- Random Forest Regressor
- XGBoost Regressor
- Used to compare performance with ANN on tabular data

---

## 📊 Model Performance Results

### 🔹 ANN Model Results
| Metric | Value |
|------|------|
| Mean Squared Error (MSE) | 2,028,927,582,208 |
| Root Mean Squared Error (RMSE) | 1,424,404 |
| Mean Absolute Error (MAE) | 1,058,153 |
| R² Score | **0.5986** |

---

### 🔹 Model Comparison (Approximate)
| Model | R² Score |
|------|---------|
| Artificial Neural Network (ANN) | ~0.60 |
| Random Forest Regressor | ~0.75 |
| XGBoost Regressor | ~0.80 |

---

## 📈 Key Observations
- House price shows strong correlation with area, bathrooms, and air conditioning
- ANN provides reasonable performance but struggles with small tabular datasets
- Tree-based ensemble models outperform ANN for this dataset

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras
- XGBoost

---

## 📌 Conclusion
This project demonstrates how EDA and proper preprocessing significantly impact model performance. While ANN achieves acceptable results, ensemble models like Random Forest and XGBoost provide superior accuracy for structured housing data.

---

