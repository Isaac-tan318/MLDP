import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    :root {
        --bg: #f7f8fc;
        --card: #ffffff;
        --text: #1e293b;
        --muted: #64748b;
        --primary: #6366f1;
        --primary-600: #4f46e5;
        --border: #e2e8f0;
        --accent: #14b8a6;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', 'Segoe UI', sans-serif;
        color: #1e293b !important;
    }

    /* Force dark text on all elements */
    .stApp, .stApp * {
        color: #1e293b;
    }

    /* Body text and labels */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown div, p, span, div {
        color: #1e293b !important;
    }

    label, .stRadio label, .stCheckbox label, .stSelectbox label,
    .stNumberInput label, .stTextInput label, .stTextArea label,
    .stSlider label, .stMultiSelect label, .stDateInput label,
    .stTimeInput label, [data-baseweb="label"] {
        color: #1e293b !important;
    }

    .stCaption, small, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
        color: #1e293b !important;
    }

    /* Radio and checkbox text */
    [data-testid="stRadio"] label, [data-testid="stRadio"] span,
    [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span,
    .stRadio > div, .stCheckbox > label > div {
        color: #1e293b !important;
    }

    /* Select boxes and dropdowns */
    [data-baseweb="select"] span, [data-baseweb="select"] div {
        color: #1e293b !important;
    }

    /* Number input */
    [data-testid="stNumberInput"] label, [data-testid="stNumberInput"] span {
        color: #1e293b !important;
    }

    /* Slider */
    [data-testid="stSlider"] label, [data-testid="stSlider"] span,
    .stSlider > div > div > div {
        color: #1e293b !important;
    }

    /* Expander */
    .stExpander summary, .stExpander summary span,
    [data-testid="stExpander"] summary span {
        color: #1e293b !important;
    }

    .stApp {
        background: radial-gradient(1200px 600px at 15% 10%, #ddd6fe 0%, transparent 55%),
                    radial-gradient(900px 500px at 85% 15%, #c7d2fe 0%, transparent 55%),
                    linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }

    [data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid var(--border);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3, h4, [data-testid="stHeadingWithActionElements"] {
        color: #1e293b !important;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 8px 18px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }

    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
    }

    .stExpander {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: #fafafa;
    }

    input, textarea, select {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    .stAlert {
        border-radius: 12px;
    }

    /* Success and error messages */
    [data-testid="stAlert"] p {
        color: inherit !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    }

    /* DataFrame text */
    .stDataFrame, .stDataFrame td, .stDataFrame th {
        color: #1e293b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("diabetes_model.pkl")

try:
    model = load_model()
except Exception:
    st.error("Model file could not be loaded. Please ensure 'diabetes_model.pkl' is available.")
    st.stop()

st.title("Diabetes Risk Predictor")
st.write("Enter your information below to estimate diabetes risk.")

with st.expander("What this tool does", expanded=False):
    st.write(
        "This app estimates diabetes risk based on general health and lifestyle inputs. "
        "It is not a medical diagnosis."
    )

age_options = [
    ("18-24", 1),
    ("25-29", 2),
    ("30-34", 3),
    ("35-39", 4),
    ("40-44", 5),
    ("45-49", 6),
    ("50-54", 7),
    ("55-59", 8),
    ("60-64", 9),
    ("65-69", 10),
    ("70-74", 11),
    ("75-79", 12),
    ("80+", 13),
]
education_options = [
    ("Never attended school or kindergarten only", 1),
    ("Grades 1-8", 2),
    ("Grades 9-11", 3),
    ("High school grad / GED", 4),
    ("College 1-3 years", 5),
    ("College 4+ years", 6),
]
income_options = [
    ("Less than $10,000", 1),
    ("$10,000-$14,999", 2),
    ("$15,000-$19,999", 3),
    ("$20,000-$24,999", 4),
    ("$25,000-$34,999", 5),
    ("$35,000-$49,999", 6),
    ("$50,000-$74,999", 7),
    ("$75,000 or more", 8),
]
gen_health_options = ["Excellent", "Very good", "Good", "Fair", "Poor"]
gen_health_map = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}

col_input, col_output = st.columns([1.2, 1])

with col_input:
    st.subheader("Step 1: Measurements")
    unit_system = st.radio(
        "Units",
        ["Metric (cm, kg)", "Imperial (ft/in, lb)"],
        horizontal=True,
        help="Choose the units you are most comfortable with.",
    )

    if unit_system.startswith("Metric"):
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
        bmi = weight_kg / ((height_cm / 100) ** 2) if height_cm > 0 else np.nan
    else:
        height_ft = st.number_input("Height (ft)", min_value=3, max_value=8, value=5, step=1)
        height_in = st.number_input("Height (in)", min_value=0, max_value=11, value=6, step=1)
        weight_lb = st.number_input("Weight (lb)", min_value=44.0, max_value=660.0, value=154.0, step=1.0)
        total_inches = (height_ft * 12) + height_in
        bmi = (weight_lb / (total_inches ** 2)) * 703 if total_inches > 0 else np.nan

    st.metric("Calculated BMI", f"{bmi:.1f}" if np.isfinite(bmi) else "N/A")

    st.subheader("Step 2: Health and lifestyle")
    col_left, col_right = st.columns(2)
    with col_left:
        high_bp = st.checkbox("High blood pressure")
        high_chol = st.checkbox("High cholesterol")
        smoker = st.checkbox("Smoked at least 100 cigarettes in lifetime")
        stroke = st.checkbox("History of stroke")
        heart_disease = st.checkbox("Heart disease or heart attack")
    with col_right:
        fruits = st.checkbox("Eats fruit daily", value=True)
        veggies = st.checkbox("Eats vegetables daily", value=True)
        hvy_alcohol = st.checkbox("Heavy alcohol consumption")
        diff_walk = st.checkbox("Difficulty walking or climbing stairs")

    st.subheader("Step 3: Overall health")
    gen_health_label = st.select_slider("General health", options=gen_health_options, value="Good")
    ment_hlth = st.number_input("Days of poor mental health (past 30 days)", min_value=0, max_value=30, value=0)
    phys_hlth = st.number_input("Days of poor physical health (past 30 days)", min_value=0, max_value=30, value=0)

    st.subheader("Step 4: Demographics")
    sex = st.radio("Sex at birth", ("Female", "Male"), horizontal=True)
    age_label = st.selectbox("Age range", [item[0] for item in age_options], index=4)
    education_label = st.selectbox("Education level", [item[0] for item in education_options], index=3)
    income_label = st.selectbox("Household income", [item[0] for item in income_options], index=4)

    errors = []
    if not np.isfinite(bmi):
        errors.append("BMI could not be calculated. Please check height and weight.")
    elif bmi < 10 or bmi > 95:
        errors.append("BMI looks outside the expected range (10-95). Please review height and weight.")

    input_data = {
        "HighBP": int(high_bp),
        "HighChol": int(high_chol),
        "BMI": float(bmi) if np.isfinite(bmi) else np.nan,
        "Smoker": int(smoker),
        "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heart_disease),
        "Fruits": int(fruits),
        "Veggies": int(veggies),
        "HvyAlcoholConsump": int(hvy_alcohol),
        "GenHlth": gen_health_map[gen_health_label],
        "MentHlth": int(ment_hlth),
        "PhysHlth": int(phys_hlth),
        "DiffWalk": int(diff_walk),
        "Sex": 1 if sex == "Male" else 0,
        "Age": dict(age_options)[age_label],
        "Education": dict(education_options)[education_label],
        "Income": dict(income_options)[income_label],
    }

    feature_order = [
        "HighBP",
        "HighChol",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income",
    ]
    input_df = pd.DataFrame([input_data], columns=feature_order)

with col_output:
    st.image("Tiny doctors.jpg", use_container_width=True)
    st.subheader("Step 5: Results")
    st.write("Results update automatically when inputs are valid.")

    if errors:
        st.error("Please fix the following before predicting:\n- " + "\n- ".join(errors))
    else:
        try:
            prediction = model.predict(input_df)
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_df)[0][1]
            else:
                probability = None
        except Exception:
            st.error("Prediction failed. Please try again or contact the app owner.")
        else:
            if prediction[0] == 1:
                st.error("High risk of diabetes")
            else:
                st.success("Low risk of diabetes")

            if probability is not None:
                safe_prob = max(0.0, min(float(probability), 1.0))
                st.progress(int(round(safe_prob * 100)))
                st.caption(f"Estimated probability: {probability:.1%}")

    with st.expander("Review your inputs"):
        st.dataframe(input_df.T, use_container_width=True)
