import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import shap
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load Dataset
file_path = r"C:\Users\Talha Saeed\PycharmProjects\week2task1\WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(file_path)

# Convert categorical variables to numeric
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop non-informative columns
df.drop(columns=["EmployeeNumber", "Over18", "EmployeeCount", "StandardHours"], inplace=True)

# Define features and target
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Convert back to DataFrame to avoid SHAP errors
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Train Random Forest Classifier with class weights and tuning
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Train Logistic Regression with class weights
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Train XGBoost Classifier
xgb_model = XGBClassifier(scale_pos_weight=5, eval_metric="logloss")
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# ROC Curve Plot
plt.figure(figsize=(10, 6))
for model, name in zip([rf_model, lr_model, xgb_model], ["Random Forest", "Logistic Regression", "XGBoost"]):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Improved SHAP Explanation
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_df)
plt.figure(figsize=(14, 8))
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test_df, feature_names=X.columns, show=False)
else:
    shap.summary_plot(shap_values, X_test_df, feature_names=X.columns, show=False)
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Actionable Insights
def provide_insights():
    print("\nActionable Insights to Reduce Attrition:")
    print("1. Salary Adjustments : Competitive compensation can improve retention.")
    print("2. Work-Life Balance : Encourage a balanced workload and flexible hours.")
    print("3. Career Growth : Providing clear career advancement paths is crucial.")
    print("4. Employee Satisfaction : Regular engagement surveys can boost morale.")
    print("5. Training & Development : Investing in employee skills reduces turnover.")

provide_insights()
