import streamlit as st
import pandas as pd
import plotly.express as px

data = pd.read_csv('Healthcare-Diabetes.csv')  

def create_figures(filtered_data):
    fig_glucose = px.histogram(filtered_data, x='Glucose', color='Outcome', nbins=20,
                                labels={'Outcome': 'Diabetes Risk'},
                                title="Distribution of Glucose Levels by Diabetes Risk")

    fig_bmi_glucose = px.scatter(filtered_data, x='BMI', y='Glucose', color='Outcome',
                                 labels={'Outcome': 'Diabetes Risk', 'BMI': 'Body Mass Index (BMI)', 'Glucose': 'Glucose Level'},
                                 title="Scatter Plot of BMI vs. Glucose by Diabetes Risk")

    fig_bmi_box = px.box(filtered_data, x='Outcome', y='BMI', color='Outcome',
                         labels={'Outcome': 'Diabetes Risk', 'BMI': 'Body Mass Index (BMI)'},
                         title="Box Plot of BMI by Diabetes Risk")

    return fig_glucose, fig_bmi_glucose, fig_bmi_box


def predict_risk(glucose, bmi):
    if glucose > 125 and bmi > 30:
        return "High risk of diabetes."
    elif glucose > 100 or bmi > 25:
        return "Moderate risk of diabetes."
    else:
        return "Low risk of diabetes."


def main():
    
    st.markdown("<h1 style='text-align: center; white-space: nowrap;'>Diabetes Risk Analysis and Preventive Measures</h1>", unsafe_allow_html=True)


    st.subheader("Select Age Range:")
    age_range = st.slider(
        'Select Age Range',
        min_value=int(data['Age'].min()),
        max_value=int(data['Age'].max()),
        value=(int(data['Age'].min()), int(data['Age'].max())),
        step=1
    )

    filtered_data = data[(data['Age'] >= age_range[0]) & (data['Age'] <= age_range[1])]

    fig_glucose, fig_bmi_glucose, fig_bmi_box = create_figures(filtered_data)

    st.plotly_chart(fig_glucose)
    st.plotly_chart(fig_bmi_glucose)
    st.plotly_chart(fig_bmi_box)

    col1, col2 = st.columns([1, 1])  

    with col1:
        st.sidebar.header('Enter Your Data for Diabetes Risk Prediction')
        glucose = st.sidebar.number_input("Enter Glucose Level:", min_value=0, max_value=300, value=100, step=1)
        bmi = st.sidebar.number_input("Enter BMI:", min_value=0.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        if st.sidebar.button('Predict Risk'):
            result = predict_risk(glucose, bmi)
            st.sidebar.success(f"Prediction: {result}")


if __name__ == '__main__':
    main()
