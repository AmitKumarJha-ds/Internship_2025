# Task 4: Regression Analysis - House Price Prediction

## 🎯 Objective
Build a regression model to predict house prices based on various features like size, location, and number of rooms using **Linear Regression**.

## 📊 Key Features
- **Size**: Numeric (e.g., in square feet).
- **Location**: Categorical (e.g., urban, suburban, rural).
- **Number of Rooms**: Numeric.
- **Price**: Numeric (target variable).

## 🛠️ Tools Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Model**: Linear Regression

## 📈 Process Overview
1. **Data Exploration**:
   - Load and explore the dataset.
   - Check for missing values and handle them appropriately.
   - Analyze distributions of numerical variables (e.g., Size, Price).
   - Identify outliers that might affect the model.

2. **Data Preprocessing**:
   - Normalize numerical data (e.g., Size and Number of Rooms).
   - Encode categorical features like **Location** using One-Hot Encoding.

3. **Feature Selection**:
   - Perform correlation analysis to identify the relationship between features and the target variable **Price**.
   - Drop low-impact predictors to improve model performance.

4. **Model Training**:
   - Split the dataset into training and testing sets (e.g., 80% train, 20% test).
   - Train a Linear Regression model using **Scikit-learn**.

5. **Model Evaluation**:
   - Calculate **Root Mean Square Error (RMSE)** for prediction accuracy.
   - Calculate **R² (Coefficient of Determination)** to assess how well the model explains variability in the data.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/amitkumarjha/main-flow-internship.git
