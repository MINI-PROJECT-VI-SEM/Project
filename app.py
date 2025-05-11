import streamlit as st
import requests
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="🌫 AQI Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with background image and styling
st.markdown(
    """
    <style>
        body {
            background-image: url("https://images.unsplash.com/photo-1580193769210-b8d1c049a7d9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2074&q=80");
            background-size: cover;
            background-attachment: fixed;
            color: #ffffff;
        }
        .main {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .stButton>button {
            background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .stTextInput>div>div>input {
            background-color: rgba(255,255,255,0.2);
            color: white;
            border-radius: 12px;
            padding: 12px;
        }
        .css-1aumxhk {
            background-color: rgba(0,0,0,0.5);
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App Header with Logo ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3579/3579166.png", width=100)
with col2:
    st.title("🌍 Air Quality Index Prediction")
    st.markdown("""
    <div style="border-left: 4px solid #6a11cb; padding-left: 12px; margin-bottom: 24px;">
        <p style="font-size: 18px; color: #e0e0e0;">Real-time air quality monitoring for Nagpur and nearby cities.</p>
    </div>
    """, unsafe_allow_html=True)

# --- API Key ---
API_KEY = "1a1a72d4d86fc5ba2c23434401a97b94"

# Coordinates for cities
CITY_COORDINATES = {
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Adilabad": {"lat": 19.68, "lon": 78.53},
    "Durg": {"lat": 21.19, "lon": 81.28},
    "Chhindwara": {"lat": 22.06, "lon": 78.94},
    "Amravati": {"lat": 20.93, "lon": 77.75}
}

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open("aqi_xgboost_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# --- Fetch Real-Time Data for a City ---
def fetch_city_data(city_name):
    if city_name not in CITY_COORDINATES:
        return None, None
    
    lat = CITY_COORDINATES[city_name]["lat"]
    lon = CITY_COORDINATES[city_name]["lon"]
    
    try:
        # Fetch weather data
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        weather_data = requests.get(weather_url).json()
        
        # Fetch air pollution data
        pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        pollution_data = requests.get(pollution_url).json()
        
        # Extract weather information
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        pressure = weather_data['main']['pressure']
        wind_speed = weather_data['wind']['speed']
        weather_desc = weather_data['weather'][0]['description'].title()
        
        # Extract air quality information
        aqi = pollution_data['list'][0]['main']['aqi'] if city_name == "Nagpur" else None
        components = pollution_data['list'][0]['components']
        
        # Prepare features for model prediction (18 features as expected by the model)
        features = [
            temp, humidity, pressure, wind_speed,
            components['co'], components['no'], components['no2'], 
            components['o3'], components['so2'], components['pm2_5'], 
            components['pm10'], components['nh3'],
            # Adding duplicate features to match the expected 18 features
            components['pm2_5'], components['pm10'], components['no2'],
            components['o3'], components['so2'], components['nh3']
        ]
        
        # Prepare display data
        display_data = {
            "Temperature (°C)": temp,
            "Humidity (%)": humidity,
            "Pressure (hPa)": pressure,
            "Wind Speed (m/s)": wind_speed,
            "Weather Condition": weather_desc,
            "AQI": aqi,
            "CO (μg/m³)": components['co'],
            "NO (μg/m³)": components['no'],
            "NO₂ (μg/m³)": components['no2'],
            "O₃ (μg/m³)": components['o3'],
            "SO₂ (μg/m³)": components['so2'],
            "PM2.5 (μg/m³)": components['pm2_5'],
            "PM10 (μg/m³)": components['pm10'],
            "NH₃ (μg/m³)": components['nh3'],
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return features, display_data
    
    except Exception as e:
        st.error(f"Error fetching data for {city_name}: {str(e)}")
        return None, None

# --- Predict AQI ---
def predict_aqi(model, features):
    try:
        # Ensure we have exactly 18 features as expected by the model
        if len(features) != 18:
            st.error(f"Expected 18 features, got {len(features)}")
            return None
            
        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]
        # Scale prediction to 1-5 range to match OpenWeatherMap AQI scale
        return min(max(1, round(prediction)), 5)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# --- AQI Information ---
def get_aqi_category(aqi):
    if aqi <= 1:
        return "Good", "#90be6d", "Air quality is satisfactory."
    elif aqi <= 2:
        return "Fair", "#f9c74f", "Air quality is acceptable."
    elif aqi <= 3:
        return "Moderate", "#f9844a", "Members of sensitive groups may experience health effects."
    elif aqi <= 4:
        return "Poor", "#f94144", "Health effects may be experienced by everyone."
    else:
        return "Very Poor", "#9d4edd", "Health alert - everyone may experience serious health effects."

# --- Display Nagpur Data (Full Details) ---
def display_nagpur_data(data):
    if not data or data['AQI'] is None:
        st.warning("Could not load Nagpur data")
        return
    
    aqi = data['AQI']
    aqi_category, aqi_color, aqi_desc = get_aqi_category(aqi)
    
    # Display Nagpur AQI
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {aqi_color} 0%, rgba(0,0,0,0.3) 100%);
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 24px;
                text-align: center;">
        <h2 style="color: white; margin-bottom: 8px;">Nagpur Air Quality</h2>
        <div style="display: flex; justify-content: center; align-items: center; gap: 24px;">
            <div style="background-color: rgba(0,0,0,0.3); 
                        border-radius: 50%; 
                        width: 120px; 
                        height: 120px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        border: 5px solid white;">
                <h1 style="color: white; margin: 0; font-size: 42px;">{aqi}</h1>
            </div>
            <div style="text-align: left;">
                <h3 style="color: white; margin: 8px 0;">{aqi_category}</h3>
                <p style="color: white; margin: 0;">{aqi_desc}</p>
                <p style="color: white; margin: 8px 0 0 0; font-size: 12px;">
                    Source: Real-time API • Last updated: {data['Last Updated']}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display weather information
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(106,17,203,0.7) 0%, rgba(37,117,252,0.7) 100%);
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 24px;">
        <h3 style="color: white; text-align: center;">🌤️ Current Weather Conditions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    weather_cols = st.columns(4)
    weather_metrics = [
        ("Temperature", f"{data['Temperature (°C)']} °C", "#6a11cb"),
        ("Humidity", f"{data['Humidity (%)']}%", "#2575fc"),
        ("Pressure", f"{data['Pressure (hPa)']} hPa", "#4cc9f0"),
        ("Wind Speed", f"{data['Wind Speed (m/s)']} m/s", "#4895ef"),
        ("Condition", data['Weather Condition'], "#4361ee")
    ]
    
    for i, (label, value, color) in enumerate(weather_metrics):
        with weather_cols[i % 4]:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {color} 0%, rgba(0,0,0,0.3) 100%);
                            padding: 16px; 
                            border-radius: 16px; 
                            color: white; 
                            text-align: center;
                            margin-bottom: 16px;">
                    <h5 style="margin: 0; font-size: 16px;">{label}</h5>
                    <h3 style="margin: 8px 0; font-size: 24px;">{value}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Display pollution components
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(106,17,203,0.7) 0%, rgba(37,117,252,0.7) 100%);
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 16px;">
        <h3 style="color: white; text-align: center;">📊 Pollution Components (μg/m³)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    pollution_cols = st.columns(4)
    pollution_metrics = [
        ("PM2.5", data['PM2.5 (μg/m³)'], "#f94144"),
        ("PM10", data['PM10 (μg/m³)'], "#f9844a"),
        ("CO", data['CO (μg/m³)'], "#f9c74f"),
        ("NO₂", data['NO₂ (μg/m³)'], "#90be6d"),
        ("O₃", data['O₃ (μg/m³)'], "#43aa8b"),
        ("SO₂", data['SO₂ (μg/m³)'], "#577590"),
        ("NH₃", data['NH₃ (μg/m³)'], "#9d4edd")
    ]
    
    for i, (label, value, color) in enumerate(pollution_metrics):
        with pollution_cols[i % 4]:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {color} 0%, rgba(0,0,0,0.3) 100%);
                            padding: 16px; 
                            border-radius: 16px; 
                            color: white; 
                            text-align: center;
                            margin-bottom: 16px;">
                    <h5 style="margin: 0; font-size: 16px;">{label}</h5>
                    <h3 style="margin: 8px 0; font-size: 24px;">{round(value, 2)}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

# --- Display Nearby City AQI (Simple View) ---
def display_nearby_city_aqi(city_name, aqi, last_updated):
    if aqi is None:
        return
    
    aqi_category, aqi_color, aqi_desc = get_aqi_category(aqi)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {aqi_color} 0%, rgba(0,0,0,0.3) 100%);
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="color: white; margin: 0 0 8px 0;">{city_name}</h3>
                <p style="color: white; margin: 0; font-size: 14px;">{aqi_category} • {aqi_desc}</p>
            </div>
            <div style="background-color: rgba(0,0,0,0.3); 
                        border-radius: 50%; 
                        width: 80px; 
                        height: 80px; 
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        border: 3px solid white;">
                <h2 style="color: white; margin: 0; font-size: 28px;">{aqi}</h2>
            </div>
        </div>
        <p style="color: white; margin: 8px 0 0 0; font-size: 12px; text-align: right;">
            Source: Model Prediction • Last updated: {last_updated}
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- AQI Information Sidebar ---
def show_aqi_info():
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, rgba(106,17,203,0.7) 0%, rgba(37,117,252,0.7) 100%);
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 16px;">
        <h3 style="color: white; text-align: center;">ℹ️ AQI Scale Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    aqi_info = [
        {"Range": "1", "Category": "Good", "Color": "#90be6d", 
         "Description": "Air quality is satisfactory with little risk to health."},
        {"Range": "2", "Category": "Fair", "Color": "#f9c74f", 
         "Description": "Air quality is acceptable; some pollutants may be a moderate concern."},
        {"Range": "3", "Category": "Moderate", "Color": "#f9844a", 
         "Description": "Members of sensitive groups may experience health effects."},
        {"Range": "4", "Category": "Poor", "Color": "#f94144", 
         "Description": "Health effects may be experienced by everyone."},
        {"Range": "5", "Category": "Very Poor", "Color": "#9d4edd", 
         "Description": "Health alert - everyone may experience serious health effects."}
    ]
    
    for info in aqi_info:
        st.sidebar.markdown(
            f"""
            <div style="background-color: {info['Color']};
                        padding: 12px;
                        border-radius: 12px;
                        margin-bottom: 12px;
                        color: white;">
                <h4 style="margin: 0 0 8px 0;">{info['Range']} - {info['Category']}</h4>
                <p style="margin: 0; font-size: 14px;">{info['Description']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Main Function ---
def main():
    show_aqi_info()
    
    model = load_model()
    if model is None:
        st.stop()

    # Center the button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔍 Get Air Quality Data", key="refresh_button"):
            with st.spinner("🌤️ Fetching real-time weather and pollution data..."):
                # Process Nagpur first (full details)
                features, display_data = fetch_city_data("Nagpur")
                if features and display_data:
                    display_nagpur_data(display_data)
                
                st.markdown("---")
                st.markdown("### Nearby Cities AQI Predictions")
                
                # Process other cities (only AQI)
                for city in [c for c in CITY_COORDINATES.keys() if c != "Nagpur"]:
                    features, display_data = fetch_city_data(city)
                    if features and display_data:
                        predicted_aqi = predict_aqi(model, features)
                        display_nearby_city_aqi(city, predicted_aqi, display_data['Last Updated'])
                
                st.markdown("---")

    # Initial load
    if "initial_load" not in st.session_state:
        with st.spinner("Loading initial data..."):
            # Process Nagpur first (full details)
            features, display_data = fetch_city_data("Nagpur")
            if features and display_data:
                display_nagpur_data(display_data)
            
            st.markdown("---")
            st.markdown("### Nearby Cities AQI Predictions")
            
            # Process other cities (only AQI)
            for city in [c for c in CITY_COORDINATES.keys() if c != "Nagpur"]:
                features, display_data = fetch_city_data(city)
                if features and display_data:
                    predicted_aqi = predict_aqi(model, features)
                    display_nearby_city_aqi(city, predicted_aqi, display_data['Last Updated'])
            
            st.markdown("---")
            
            st.session_state.initial_load = True

# --- Run App ---
if __name__ == "__main__":
    main()