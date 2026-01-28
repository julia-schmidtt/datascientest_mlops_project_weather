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
            st.success("Weather Prediction API is running")
            data = response.json()
            st.json(data)
        else:
            st.error("Weather Prediction API is not responding correctly")
    except Exception as e:
        st.error(f"API is not reachable: {str(e)}")
        st.info("Make sure the Weather Prediction API is running on port 8000 and your IP address is noted in .env (see README)")

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
            st.markdown(description)
    
    if i + 1 < len(endpoints_without_button):
        with col2:
            method, path, description = endpoints_without_button[i + 1]
            
            with st.container(border=True):
                st.markdown(f"**{method} {path}**")
                st.code(f"http://localhost:8000{path}", language="text")
                st.markdown(description)
