import streamlit as st

st.title("Dataset")

tab1, tab2, tab3 = st.tabs(["Exploration", "Preprocessing", "Modeling"])

with tab1:
    st.markdown("""Insights from first dataset exploration. """)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Info:**")
            st.markdown("""
            - **Source:** Kaggle Weather Dataset [1]
            - **Duration:** Contains approximately 10 years of daily weather observations
            - **Locations:** Multiple weather stations across Australia
            """)
    with col2:
        with st.container(border=True):
            st.markdown("**First Insights:**")
            st.markdown("""
            - **Target Variable:** RainTomorrow (Binary: Yes/No)
            - **Features:** 22 (Temperature, humidity, wind, pressure, rainfall, ...)
            - **Samples**: 145460
            - **Duplicates**: No
            - **Missing Values**: >10%
            - **Class Imbalance**: >75% No Rain Tomorrow, <22% Rain Tomorrow
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

    content = "Raw dataset needed to be preprocessed before decisions regarding model type and procedure could be made."

    st.markdown(text_card.format(content=content), unsafe_allow_html=True)

with tab2:
    st.markdown("""Identification and implementation of necessary preprocessing steps.""")

    for i, (title, content) in enumerate([
        ("Step 1: Handling of Missing Values Before Data Split", """
        - **Option A**: Delete rows containing missing values for certain variables -> e.g. missing value in target variable (rows containing no target variable are useless)
        - **Option B**: Delete column -> columns containing >40% missing values
        """),
        ("Step 2: Check Data Types", """
        - Check column types and change to appropriate format
        - Date column: change from object to datetime format
        """),
        ("Step 3: Split Dataset", """
        - **Training data**: 80%
        - **Test data**: 20%
        """),
        ("Step 4: Handling of Missing Values After Data Split", """
        - **Option C**: Replace missing values with median/modus of training data -> columns containing max. 10% missing values
        - **Numerical Columns**: Replace missing values with median
        - **Categorical Columns**: Replace missing values with modus
        - **Save Median/Modus**: Save values for every column for later use
        """),
        ("Step 5: One-Hot Encoding", """
        - **Categorical Columns**: Only numerical values for easier handling
        """),
        ("Step 6: Scaling", """
        - **StandardScaler**: Use scaler for training data and apply to test data to prevent data leakage
        - **Save Scaler**: Save StandardScaler for later use
        """),
        ("Step 7: Balance Class Distribution", """
        - **Class Distribution After Preprocessing**: No Rain 77.8%, Rain 22.2%
        - **SMOTE**: Applied on training data for balanced class distribution
        """)
    ], 1):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(content)

with tab3:
    st.markdown("""Identification of best suited model and choice of hyperparameters.""")

    for j, (title2, content2) in enumerate([
        ("Step 1: Model Choice", """
        - **LazyClassifier**: Applied on SMOTE training data to test different models with default parameters
        - **Tested Models**: RandomForestClassifier, LogisticRegression, XGBClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier
        - **Best Model**: XGBClassifier
        """),
        ("Step 2: Hyperparameter Identification", """
        - **GridSearch**: Test every combination of given parameters and identify best combination
        """)
    ], 1):
        with st.container(border=True):
            st.markdown(f"**{title2}**")
            st.markdown(content2)


    text_card2 = """
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

    content3 = "Preprocessing steps, model type and model hyperparameters identified."

    st.markdown(text_card2.format(content=content3), unsafe_allow_html=True)

st.markdown("---")

with st.expander("References"):
    st.markdown("""
    [1] https://www.kaggle.com/jsphyg/weather-dataset-rattle-package, 28.01.2026
    """)
