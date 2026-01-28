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
