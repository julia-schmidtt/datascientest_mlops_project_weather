import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("API Demo")

# Get API URL from .env
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

# API Status Check
with st.container(border=True):
    st.markdown("**API Status**")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.markdown("The Weather Prediction API is running on Port 8000. Explore the available endpoints below.")
            st.success("API is running")
#            data = response.json()
#            st.json(data)
        else:
            st.error("Weather Prediction API is not responding correctly.")
    except Exception as e:
        st.error(f"API is not reachable: {str(e)}")
        st.info("Make sure the Weather Prediction API is running on port 8000 and your IP address is noted in .env (see README).")

st.markdown("---")

st.header("API Endpoints")

st.markdown("""
    <style>
    /* Alle Boxen auf gleiche Mindesthöhe setzen */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        min-height: 250px;
    }
    </style>
    """, unsafe_allow_html=True)

endpoints_with_button = [
    ("GET", "/", "Returns API information and available endpoints"),
    ("GET", "/health", "Health check endpoint"),
    ("GET", "/model/info", "Returns information regarding current production model"),
    ("GET", "/metrics", "Metrics"),
]

endpoints_without_button = [
    ("POST", "/model/refresh", "Reloads production model"),
    ("POST", "/train", "Train model for specific training data split"),
    ("POST", "/predict/simple", "Prediction of current production model, 5 required inputs (location, date, min_temp, max_temp, rain_today)"),
    ("POST", "/predict", "Prediction of current production model, 110 required inputs"),
    ("POST", "/pipeline/next-split", "Automated pipeline for triggering training with next training data split"),
    ("POST", "/pipeline/next-split-drift-detection", "Automated pipeline for triggering conditional training with next training data split if data drift is present"),
]


for i in range(0, len(endpoints_with_button), 2):
    col1, col2 = st.columns(2)
    
    with col1:
        method, path, description = endpoints_with_button[i]
        
        with st.container(border=True):
            st.markdown(f"**{method} {path}**")
            st.code(f"http://localhost:8000{path}", language="text")
            st.markdown(description)
            
            button_clicked = st.button("Try it", key=f"btn_{path}")
        
        if button_clicked:
            try:
                response = requests.get(f"{API_URL}{path}", timeout=5)
                
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'application/json' in content_type:
                        st.json(response.json())
                    else:
                        st.code(response.text, language="text")
                else:
                    st.error(f"Error {response.status_code}")
                    st.code(response.text)
            except Exception as e:
                st.error(str(e))
    
    if i + 1 < len(endpoints_with_button):
        with col2:
            method, path, description = endpoints_with_button[i + 1]
            
            with st.container(border=True):
                st.markdown(f"**{method} {path}**")
                st.code(f"http://localhost:8000{path}", language="text")
                st.markdown(description)
                
                button_clicked = st.button("Try it", key=f"btn_{path}")
            
            if button_clicked:
                try:
                    response = requests.get(f"{API_URL}{path}", timeout=5)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '')
                        
                        if 'application/json' in content_type:
                            st.json(response.json())
                        else:
                            st.code(response.text, language="text")
                    else:
                        st.error(f"Error {response.status_code}")
                        st.code(response.text)
                except Exception as e:
                    st.error(str(e))

st.markdown("---")


for i in range(0, len(endpoints_without_button), 2):
    col1, col2 = st.columns(2)
    
    with col1:
        method, path, description = endpoints_without_button[i]
        
        with st.container(border=True):
            st.markdown(f"**{method} {path}**")
            st.code(f"http://localhost:8000{path}", language="text")
            st.markdown(description)\
    
    if i + 1 < len(endpoints_without_button):
        with col2:
            method, path, description = endpoints_without_button[i + 1]
            
            with st.container(border=True):
                st.markdown(f"**{method} {path}**")
                st.code(f"http://localhost:8000{path}", language="text")
                st.markdown(description)


st.markdown("---")

st.header("Live Prediction Demo")

st.markdown("Try out the weather prediction model with custom input data. The simple prediction requires only 5 inputs.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        location = st.selectbox(
            "Location",
            ["Sydney", "Berlin", "Melbourne", "Brisbane", "Perth", "Adelaide", "Hobart", "Canberra", "Darwin"]
        )
        date = st.date_input("Date", value=None)
    
    with col2:
        min_temp = st.number_input("Min Temperature (°C)", value=10.0, step=0.1)
        max_temp = st.number_input("Max Temperature (°C)", value=20.0, step=0.1)
    
    with col3:
        rain_today = st.selectbox("Rain Today", ["No", "Yes"])
    
    submit = st.form_submit_button("🌦️ Predict Rain Tomorrow", use_container_width=True)
    
    if submit:
        # Validierung
        if date is None:
            st.error("Please select a date")
        elif min_temp >= max_temp:
            st.error("Min temperature must be less than max temperature")
        else:
            # Mapping: "Yes" -> 1, "No" -> 0
            rain_today_value = 1 if rain_today == "Yes" else 0
            
            payload = {
                "location": location,
                "date": str(date),
                "min_temp": min_temp,
                "max_temp": max_temp,
                "rain_today": rain_today_value
            }
            
            try:
                with st.spinner("Making prediction..."):
                    response = requests.post(
                        f"{API_URL}/predict/simple",
                        json=payload,
                        timeout=10
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Prediction successful!")
                    
                    # Mapping: API gibt 0 oder 1 zurück
                    prediction_value = result.get("prediction", 0)
                    prediction = "Yes" if prediction_value == 1 else "No"
                    
                    # Ergebnisse anzeigen
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.container(border=True):
                            if prediction == "Yes":
                                st.markdown("### 🌧️ Rain Expected")
                            else:
                                st.markdown("### ☀️ No Rain Expected")
                            st.metric(
                                label="Prediction",
                                value=prediction
                            )
                    
                    with col2:
                        with st.container(border=True):
                            prob_rain = result.get('probability_rain', 0) * 100
                            st.metric(
                                label="Rain Probability",
                                value=f"{prob_rain:.1f}%"
                            )
                    
                    # Request/Response Details
                    with st.expander("View Request & Response Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Request:**")
                            st.json(payload)
                        with col2:
                            st.markdown("**Response:**")
                            st.json(result)
                            
                else:
                    st.error(f"Error: HTTP {response.status_code}")
                    st.code(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("Request timeout - API took too long to respond")
            except requests.exceptions.ConnectionError:
                st.error("Connection error - Cannot reach API")
            except Exception as e:
                st.error(f"Error: {str(e)}")
