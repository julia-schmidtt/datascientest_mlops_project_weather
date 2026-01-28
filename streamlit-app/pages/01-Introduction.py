import streamlit as st

st.title("Introduction")

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**Context:**")
        st.markdown("""
        - Weather prediction based on ML model trained with weather data
        - Based on input variables: Will it rain tomorrow? -> binary classification 
        """)
with col2:
    with st.container(border=True):
        st.markdown("**Project Goals:**")
        st.markdown("""
        - Implement MLOps architecture around an API for weather prediction in Australia
        - Simulate data evolution over time
        - Model (re)training (triggered by data drift detection, scheduled, versioning, ...)
        - Visualization of API status, resource usage, ...
        - API prediction requests with current production model: Will it rain tomorrow at certain location in Australia based on certain input features?  
        """)
