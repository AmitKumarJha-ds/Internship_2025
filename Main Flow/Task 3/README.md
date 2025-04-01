# Student Pass/Fail Prediction (Classification)

## 🎯 Objective
The goal of this task is to predict whether a student will pass or fail based on their study hours and attendance.

## 📊 Key Features
- **Study Hours**: Number of hours a student studies per week.
- **Attendance**: Percentage of classes attended by the student.
- **Pass**: Binary column indicating whether the student passed (1) or failed (0).

## 🛠️ Tools Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn
- **Model**: Logistic Regression

## 📈 Process Overview
1. **Data Exploration**:
   - Check for missing values or outliers.
   - Visualize the relationship between Study Hours, Attendance, and Pass status.
   
2. **Model Training**:
   - Split the dataset into training and testing sets (80-20 split).
   - Train a logistic regression model to predict the Pass/Fail status using Study Hours and Attendance.
   
3. **Model Evaluation**:
   - Evaluate the model using:
     - **Accuracy**: The proportion of correct predictions.
     - **Confusion Matrix**: Breakdown of True Positives, True Negatives, False Positives, and False Negatives.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/amitkumarjha/main-flow-internship.git
