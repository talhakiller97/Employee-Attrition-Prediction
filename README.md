
A machine learning project using the IBM HR Analytics dataset to predict employee attrition and identify the most influential factors. The goal is to help HR professionals reduce turnover and enhance employee satisfaction.

---

## 📌 Project Objectives

- Predict whether an employee is likely to leave the company.
- Handle class imbalance using SMOTE.
- Train and evaluate three ML models.
- Interpret model predictions using SHAP.
- Generate data-driven recommendations to reduce attrition.

---

## 📁 Dataset

- **Name:** IBM HR Analytics Employee Attrition & Performance  
- **Format:** CSV  
- **Source:** Provided in project directory  
- **File Used:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`

---

## 🧰 Tools & Libraries

- **Programming Language:** Python  
- **Libraries:**
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`, `imblearn`, `shap`

---

## 🧪 Machine Learning Models

Three classifiers were trained and compared:

| Model               | Highlights                               |
|---------------------|-------------------------------------------|
| 🎯 Random Forest       | Tuned with `class_weight=balanced`       |
| 🧮 Logistic Regression | Handles imbalance, suitable for baseline |
| ⚡ XGBoost             | Boosted performance, interpretable via SHAP |

---

## 📊 Evaluation Metrics

Each model was evaluated using:

- ✅ Accuracy  
- 📉 Confusion Matrix  
- 📈 ROC-AUC Curve  
- 🔍 Classification Report  
- 📊 SHAP Summary Plot (for XGBoost)

---

## 📉 Class Imbalance Handling

- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the `Attrition` target variable.

---

## 📈 SHAP Visualization

Feature importance from the XGBoost model was interpreted using SHAP:

```python
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_df)
shap.summary_plot(shap_values, X_test_df)


💡 Key Insights & Recommendations
To reduce employee attrition, companies should:

💰 Salary Adjustments: Offer competitive compensation packages.

🕓 Work-Life Balance: Promote flexible work arrangements.

🚀 Career Growth: Provide clear advancement opportunities.

🧠 Engagement: Conduct regular satisfaction surveys.

📚 Training Programs: Invest in employee development.

🗂️ Project Structure
graphql
Copy
Edit
├── employee_attrition_prediction.py        # Main script
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Dataset
├── shap_summary_plot.png                   # SHAP feature importance plot
└── README.md                               # This documentation file
▶️ How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/talhakiller97/employee-attrition-prediction.git
cd employee-attrition-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python employee_attrition_prediction.py
👤 Author
Talha Saeed
📍 Data Scientist
🔗 GitHub Profile
