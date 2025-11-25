# test_load_model_final.py
import joblib
import pandas as pd
import sklearn

# چاپ نسخه sklearn
print("Sklearn version in this env:", sklearn.__version__)

# --- لود مدل ---
model = joblib.load("titanic_app_protocol5.joblib")
print("Model loaded successfully! Type:", type(model))

# --- نمونه تست ---
# ترتیب ستون‌ها که pipeline انتظار دارد
columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']

# یک نمونه فرضی
X_test = pd.DataFrame([[3, 22, 1, 0, 7.25, 'male', 'S']], columns=columns)

# --- پیش‌بینی ---
prediction = model.predict(X_test)
print("Prediction for test sample:", prediction)

# --- پیش‌بینی احتمال (اگر مدل این قابلیت را داشته باشد) ---
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_test)
    print("Prediction probability:", proba)
