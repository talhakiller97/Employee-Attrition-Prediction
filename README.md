🧠 Employee Attrition Prediction
This project applies machine learning techniques to predict employee attrition using IBM’s HR Analytics dataset. It aims to uncover the key drivers behind employee turnover and offer actionable recommendations for HR decision-makers.



📂 Dataset
Name: IBM HR Analytics Employee Attrition & Performance

File: WA_Fn-UseC_-HR-Employee-Attrition.csv

🚀 Features
End-to-end ML pipeline: preprocessing → modeling → evaluation

Class imbalance handled via SMOTE

Three models trained: Random Forest, Logistic Regression, XGBoost

Feature importance visualized with SHAP

Insights generated to improve employee retention

🛠️ Tools & Libraries
Python 3.x

pandas, numpy, matplotlib, seaborn

scikit-learn, xgboost, imblearn, shap

🧪 Models & Evaluation
Model	Accuracy	Highlights
Random Forest	✅	Tuned with class weights
Logistic Regression	✅	Balanced class support
XGBoost	✅	Optimized with scale_pos_weight
Evaluation metrics:

Accuracy Score

Confusion Matrix

ROC-AUC Curve

SHAP Summary Plot

📈 SHAP Analysis
Feature contributions are visualized using SHAP to interpret the XGBoost model:

python
Copy
Edit
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_df)
shap.summary_plot(shap_values, X_test_df)
💡 Key Recommendations
Actionable insights based on model predictions:

💰 Salary Adjustments: Ensure competitive compensation

🕓 Work-Life Balance: Encourage flexible work hours

📈 Career Growth: Provide clear promotion pathways

🧠 Employee Satisfaction: Use engagement surveys regularly

🎯 Upskilling: Invest in training and development programs

📁 Project Structure
Copy
Edit
├── employee_attrition_prediction.py
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── shap_summary_plot.png
└── README.md
✅ How to Run
Clone the repo

bash
Copy
Edit
git clone https://github.com/talhakiller97/employee-attrition-prediction.git
cd employee-attrition-prediction
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the project

bash
Copy
Edit
python employee_attrition_prediction.py
👤 Author
Talha Saeed
📧 GitHub
