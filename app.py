import streamlit as st
import pandas as pd
import joblib
import json


rf_model = joblib.load('new_random_forest_model.joblib')
scaler = joblib.load('new_scaler.pkl')

with open('new_feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

city_columns = joblib.load('new_city_columns.pkl')


st.set_page_config(page_title="PM2.5 Prediction App", layout="wide")

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background-color: #f8f9fa; 
    }

    /* Optional: change font color for all text */
    .stApp * {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("PM2.5 Prediction App")


def interpret_pm25(pm):
    if pm <= 12:
        return "Good: Air quality is satisfactory, little or no risk.", "#4CAF50"
    elif pm <= 35.4:
        return "Moderate: Acceptable air quality; some sensitive people may be affected.", "#FFEB3B"
    elif pm <= 55.4:
        return "Unhealthy for Sensitive Groups: People with respiratory issues should reduce outdoor activity.", "#FF9800"
    elif pm <= 150.4:
        return "Unhealthy: Everyone may begin to experience health effects; sensitive groups more affected.", "#F44336"
    elif pm <= 250.4:
        return "Very Unhealthy: Health alert; everyone may experience more serious effects.", "#9C27B0"
    else:
        return "Hazardous: Emergency conditions; entire population may be affected.", "#7B1FA2"


uploaded_file = st.file_uploader("Upload Excel file for batch prediction", type=["xlsx", "xls"])


def predict_pm25(df_input):
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]
    scaled_input = scaler.transform(df_input)
    predictions = rf_model.predict(scaled_input)
    return predictions

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    predictions = predict_pm25(df_input)
    df_results = pd.DataFrame({
        'Predicted_PM2.5': predictions,
        'Interpretation': [interpret_pm25(pm)[0] for pm in predictions]
    })


    st.subheader("Batch Predictions")

    for i, pm in enumerate(predictions):
        interpretation_text, color = interpret_pm25(pm)

        st.markdown(f"""
        <div style="
        padding:20px;
        border-radius:12px;
        background-color:{color};
        color:white;
        text-align:center;
        margin-top:15px;
        box-shadow:0 4px 10px rgba(0,0,0,0.15);
        ">
        <h3>Row {i + 1}</h3>
        <h2>Predicted PM2.5: {pm:.2f} µg/m³</h2>
        <p>{interpretation_text}</p>
        </div>
     """, unsafe_allow_html=True)




else:
    with st.container():
        st.subheader("Manual Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            main_aqi = st.slider("AQI", 0.0, 500.0, 2.0, key="aqi")
            components_co = st.slider("CO", 0.0, 500.0, 283.72, key="co")
            components_no = st.slider("NO", 0.0, 50.0, 0.01, key="no")
            components_no2 = st.slider("NO2", 0.0, 200.0, 2.55, key="no2")
            components_o3 = st.slider("O3", 0.0, 300.0, 49.35, key="o3")
            components_so2 = st.slider("SO2", 0.0, 100.0, 1.31, key="so2")
            
        with col2:
            components_pm10 = st.slider("PM10", 0.0, 500.0, 6.95, key="pm10")
            components_nh3 = st.slider("NH3", 0.0, 50.0, 1.46, key="nh3")
            hour = st.slider("Hour", 0, 23, 12, key="hour")
            day_of_week = st.slider("Day of Week (0=Monday)", 0, 6, 2, key="day")
            is_weekend = 1 if st.selectbox("Is it a weekend?", ["No", "Yes"], key="weekend") == "Yes" else 0
            selected_city = st.selectbox("Select City", [c.replace("city_name_", "") for c in city_columns], key="city")
    

    default_input = {
        'main.aqi': main_aqi,
        'components.co': components_co,
        'components.no': components_no,
        'components.no2': components_no2,
        'components.o3': components_o3,
        'components.so2': components_so2,
        'components.pm10': components_pm10,
        'components.nh3': components_nh3,
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'month': 1
    }
    
    df_input = pd.DataFrame([default_input])
    
    
    for city_col in city_columns:
        df_input[city_col] = 0
    city_col_name = f"city_name_{selected_city}"
    if city_col_name in df_input.columns:
        df_input[city_col_name] = 1
    
    
    pred_pm25 = predict_pm25(df_input)[0]
    interpretation_text, color = interpret_pm25(pred_pm25)
    
    
    with st.container():
        st.markdown(f"""
        <div style="padding:20px; border-radius:12px; background-color:{color}; color:white; text-align:center; margin-top:20px;">
            <h2>Predicted PM2.5: {pred_pm25:.2f} µg/m³</h2>
            <p>{interpretation_text}</p>
        </div>
        """, unsafe_allow_html=True)
