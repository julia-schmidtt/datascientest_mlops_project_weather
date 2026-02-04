import streamlit as st

st.title("Introduction")

tab1, = st.tabs(["Project Goals & Context"])

with tab1:
    with st.container(border=True):
        st.markdown("""
        - Implement automated MLOps architecture around an API for weather prediction in Australia
        - Prediction made by ML model trained with weather data

        - Simulate data evolution over time
        - Model training (triggered by e.g. data drift detection)

        - Visualization of API status, resource usage, ...
        """)

    text_card = """
    <div style="
        background-color: #4090c9;
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        font-size: 16px;
        font-weight: 500;
        margin: 20px 0;
    ">
    {content}
    </div>
    """

    content = "Build MLOps architecture to answer central question: Will it rain tomorrow?"

    st.markdown(text_card.format(content=content), unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        "/app/images/architecture.png",
        width=600
    )
