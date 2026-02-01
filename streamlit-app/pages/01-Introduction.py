import streamlit as st

st.title("Introduction")

tab1, = st.tabs(["Project Goals & Context"])

with tab1:
    with st.container(border=True):
        st.markdown("""
        - Implement MLOps architecture around an API for weather prediction in Australia
        - Question based on input variables: **Will it rain tomorrow?**
        - Prediction made by ML model trained with weather data

        - Simulate data evolution over time
        - Model (re)training (triggered by data drift detection)

        - Visualization of API status, resource usage, ...
        """)
