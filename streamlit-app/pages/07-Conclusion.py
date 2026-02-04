import streamlit as st

st.title("Outlook and Conclusion")

tab1, tab2 = st.tabs(["Further Improvements", "Conclusion"])

with tab1:
    with st.container(border=True):
        st.markdown("""
        - Runtime optimization of the API endpoints
            - optimize code
            - divide API functions into different Docker containers
        - Orchestration:
            - usage of tools like Kubernetes to improve API downtime and accessibility
            - scalability
        - Security optimization: 
            - definition of user groups with limited access to the API endpoints
            - e.g. reverse proxy
        """)

with tab2:
    with st.container(border=True):
        st.markdown("""
        - Application of content of the lessions
        - Exploration of different tools
        - Building MLOps system: complex task
        - Opportunity to apply different tools in a given context
        """)
