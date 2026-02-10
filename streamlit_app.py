import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

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
    st.image("Tiny doctors.jpg", width=160)
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
