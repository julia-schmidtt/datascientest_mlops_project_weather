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
        else:
            st.error("Weather Prediction API is not responding correctly.")
    except Exception as e:
        st.error(f"API is not reachable: {str(e)}")
        st.info("Make sure the Weather Prediction API is running on port 8000 and your IP address is noted in .env (see README).")

st.markdown("---")

tab1, tab2 = st.tabs(["API Endpoints", "Live Weather Prediction"])

with tab1:
    with st.container(border=True):
        st.markdown("**POST/login**")
        st.code("http://localhost:8000/login", language="text")
        st.markdown("First step: Use login endpoint to get token for API security. Use token for other endpoints.")

    with st.expander("Example Usage"):
        st.markdown("1. Get token via login endpoint:")
        st.code("""curl -X 'POST' \\
      'http://localhost:8000/login' \\
      -H 'accept: application/json' \\
      -H 'Content-Type: application/json' \\
      -d '{
      "username": "admin",
      "password": "admin"
    }'""", language="bash")
        st.markdown("Returns Token, e.g.: 1234")

        st.markdown("2. Use token for endpoints:")
        st.code("""curl -X 'GET' \\
      'http://localhost:8000/model/info' \\
      -H 'accept: application/json' \\
      -H 'Authorization: Bearer 1234'""", language="bash")

        st.code("""curl -X POST http://localhost:8000/predict/simple \\
  -H "Content-Type: application/json" \\
  -H 'Authorization: Bearer 1234' \\
  -d '{
    "location": "Sydney",
    "date": "2025-01-15",
    "min_temp": 18.0,
    "max_temp": 28.0,
    "rain_today": 0
  }' """, language="bash")

        st.code("""curl -X POST http://localhost:8000/pipeline/next-split \\
  -H "Content-Type: application/json" \\
  -H 'Authorization: Bearer 1234'  """, language="bash")

        st.code("""curl -X POST http://localhost:8000/pipeline/next-split-drift-detection \\
  -H "Content-Type: application/json" \\
  -H 'Authorization: Bearer 1234'  """, language="bash")


    st.markdown("---")
 

    st.markdown("""
        <style>
        /* all boxes min height */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 250px;
        }
        </style>
        """, unsafe_allow_html=True)

    endpoints_with_button = [
        ("GET", "/", "Returns API information and available endpoints."),
        ("GET", "/health", "Health check endpoint."),
        ("GET", "/model/info", "Returns information regarding current production model."),
        ("GET", "/metrics", "Returns API metrics."),
    ]

    endpoints_without_button = [
        ("POST", "/model/refresh", "Reloads current production model."),
        ("POST", "/train", "Manually train model for specific training data split."),
        ("POST", "/predict/simple", "Prediction for 'Rain Tomorrow' of current production model, 5 inputs required (location, date, min_temp, max_temp, rain_today)."),
        ("POST", "/predict", "Prediction of 'Rain Tomorrow' of current production model, 110 inputs required."),
        ("POST", "/pipeline/next-split", "Automated pipeline for triggering data growth, model training and new decision regarding production model."),
        ("POST", "/pipeline/next-split-drift-detection", "Automated pipeline for triggering data growth, data drift detection, conditional training if data drift is detected and decision regarding production model. If no data drift is present, no training will be performed and the current production model will not change."),
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

with tab2:

    st.markdown("Try out the weather prediction API for locations in Australia. Here the predict/simple endpoint is demonstrated, therefore only 5 inputs are required.")

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
            #  Validation
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

                        # Mapping
                        prediction_value = result.get("prediction", 0)
                        prediction = "Yes" if prediction_value == 1 else "No"

                        # Show results
                        col1, col2 = st.columns(2)

                        with col1:
                            with st.container(border=True):
                                if prediction == "Yes":
                                    st.markdown("### 🌧️  Rain Expected Tomorrow")
                                else:
                                    st.markdown("### ☀️ No Rain Expected Tomorrow")

                        with col2:
                            with st.container(border=True):
                                prob_rain = result.get('probability_rain', 0) * 100
                                st.metric(
                                    label='Model Prediction',
                                    value=prediction
                                )
                                st.metric(
                                    label="Rain Probability for Tomorrow",
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

