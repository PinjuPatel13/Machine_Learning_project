import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("updated_data.csv")

def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['Churn'] = label_encoder.fit_transform(df['Churn']) 
    df = df.select_dtypes(include=['float64', 'int64']) 
    return df

df_cleaned = preprocess_data(df)
X = df_cleaned.drop('Churn', axis=1)
y = df_cleaned['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg) * 100
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
accuracy_xgb = accuracy_score(y_test, y_pred_xgb) * 100

st.markdown("<h1 style='text-align: center; white-space: nowrap;'>Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)


st.sidebar.header('Input Customer Data')
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.sidebar.number_input('Tenure (in months)', min_value=0, max_value=100, value=12)
monthly_charges = st.sidebar.number_input('Monthly Charges', min_value=20, max_value=200, value=50)

inputs = np.array([age, tenure, monthly_charges]).reshape(1, -1)

prediction_logreg = logreg.predict(inputs)
prediction_rf = rf.predict(inputs)
prediction_xgb = xgb.predict(inputs)

col1, col2 = st.columns(2)

with col1:
    st.subheader('Prediction Results')

    prediction_dict = {
        'Logistic Regression': 'Churn' if prediction_logreg[0] == 1 else 'Stay',
        'Random Forest': 'Churn' if prediction_rf[0] == 1 else 'Stay',
        'XGBoost': 'Churn' if prediction_xgb[0] == 1 else 'Stay'
    }

    for model, prediction in prediction_dict.items():
        st.markdown(f"**{model}:** {prediction}")


with col2:
    st.subheader('Model Accuracy')
    
    st.markdown(f"**Logistic Regression Accuracy:** {accuracy_logreg:.2f}%")
    st.markdown(f"**Random Forest Accuracy:** {accuracy_rf:.2f}%")
    st.markdown(f"**XGBoost Accuracy:** {accuracy_xgb:.2f}%")
    
    accuracy_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [accuracy_logreg, accuracy_rf, accuracy_xgb]
    }
    accuracy_df = pd.DataFrame(accuracy_data)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Accuracy', y='Model', data=accuracy_df, ax=ax, palette='Blues')
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    
    st.pyplot(fig)

st.subheader('Actionable Recommendations')
if any(pred == 'Churn' for pred in prediction_dict.values()):
    st.write("Recommendation: This customer is at high risk of churn. Consider offering special promotions or discounts to retain them.")
else:
    st.write("Recommendation: This customer is not at risk of churn. Continue with regular engagement strategies.")
    
    
    
