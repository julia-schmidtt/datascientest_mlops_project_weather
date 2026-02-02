import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Weather Prediction in Australia",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <div style='text-align: center;'>
        <h2>Welcome to the MLOps Weather Prediction Project!</h2>
        <p style='font-size:18px;'>
        This application presents the steps completed in this project.<br>
        </p>
        <p style='font-size:18px;'>
        Use the sidebar on the left to navigate through the different sections.<br>
       </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("##### Team")

card_style = """
<div style="
    background-color: #4090c9;
    padding: 40px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
    font-size: 24px;
    font-weight: bold;
    margin: 10px;
">
{name}
</div>
"""

team = ["Alex", "Jonas", "Kiki", "Julia"]

# 2x2 Layout
for i in range(0, len(team), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(team):
            col.markdown(card_style.format(name=team[i+j]), unsafe_allow_html=True)


st.markdown("---")


st.markdown("##### Project")

st.link_button("View on GitHub", "https://github.com/julia-schmidtt/datascientest_mlops_project_weather/tree/master?tab=readme-ov-file", use_container_width=True)
