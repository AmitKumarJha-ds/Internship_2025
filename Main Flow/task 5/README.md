# Task 5: Classification Task - Student Pass/Fail Prediction

## 🎯 Objective
Predict whether a student will pass or fail based on their **study hours** and **attendance**.

## 📊 Key Features
- **Study Hours**: Number of hours a student studies per week (Numeric).
- **Attendance**: Percentage of classes attended (Numeric).
- **Pass**: Binary target variable (1 for pass, 0 for fail).

## 🛠️ Tools Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Model**: Logistic Regression

## 📈 Process Overview
1. **Data Exploration**:
   - Load and inspect the dataset.
   - Check for missing values or outliers.
   - Visualize the relationship between **Study Hours**, **Attendance**, and **Pass** to observe any trends.
   
2. **Model Training**:
   - Split the dataset into training and testing sets (e.g., 80% train, 20% test).
   - Train a **Logistic Regression** model using **Study Hours** and **Attendance** as the features and **Pass** as the target variable.

3. **Model Evaluation**:
   - **Accuracy**: Proportion of correctly classified instances.
   - **Confusion Matrix**: Breakdown of **True Positives**, **True Negatives**, **False Positives**, and **False Negatives**.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/amitkumarjha/main-flow-internship.git
