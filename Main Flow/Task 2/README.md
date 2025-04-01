# House Price Prediction (Regression)

## 🎯 Objective
The goal of this task is to predict house prices based on features such as size, location, and number of rooms using linear regression.

## 📊 Key Features
- **Size**: Numeric (e.g., in square feet).
- **Location**: Categorical (e.g., urban, suburban, rural).
- **Number of Rooms**: Numeric.
- **Price**: Numeric (target variable).

## 🛠️ Tools Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Model**: Linear Regression

## 📈 Process Overview
1. **Load and Explore**:
   - Inspect the dataset for missing values.
   - Visualize distributions of numerical variables (e.g., Size, Price).
   - Identify potential outliers.
   
2. **Data Preprocessing**:
   - Normalize numerical data (Size, Number of Rooms).
   - Encode categorical features (Location) using one-hot encoding.
   
3. **Model Training**:
   - Split data into training and testing sets (80-20 split).
   - Train a linear regression model using Scikit-learn.
   
4. **Model Evaluation**:
   - Use RMSE (Root Mean Squared Error) and R² (Coefficient of Determination) for evaluation.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/amitkumarjha/main-flow-internship.git
