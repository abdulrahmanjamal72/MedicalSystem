import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# --- Hugging Face Setup ---
REPO_ID = "abdulrahmanjamal72/MedicalSystem" 
# We only need the optimized Random Forest models now!
ARTIFACTS = ['our_models.pkl'] 

@st.cache_resource
def load_artifacts():
    artifacts_path = {}
    for artifact in ARTIFACTS:
        file_path = hf_hub_download(repo_id=REPO_ID, filename=artifact)
        artifacts_path[artifact] = joblib.load(file_path)
    return artifacts_path

# --- UI Header ---
st.set_page_config(page_title="AI Medical Routing", layout="wide")
st.title("🏥 AI Medical Symptom Routing Dashboard")
st.write("تطبيق ذكاء اصطناعي لتوقع الأمراض وتوجيه المريض للتخصص المناسب.")

# --- Load Models ---
try:
    models_dict = load_artifacts()['our_models.pkl']
    st.sidebar.success("تم تحميل النماذج بنجاح! 🚀")
except Exception as e:
    st.error(f"خطأ في التحميل: {e}")
    st.stop()

# --- Patient Input Form ---
st.header("بيانات المريض (Patient Data)")

# Using columns and expanders to make the UI look clean and professional
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Vitals")
    age = st.number_input("Age (سن المريض)", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, value=75.0)
    systolic_bp = st.number_input("Systolic BP", min_value=70.0, max_value=250.0, value=120.0)
    diastolic_bp = st.number_input("Diastolic BP", min_value=40.0, max_value=150.0, value=80.0)

with col2:
    st.subheader("Lab Results")
    hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
    cholesterol = st.number_input("Total Cholesterol", min_value=50.0, max_value=600.0, value=200.0)
    hdl = st.number_input("HDL (Good Cholesterol)", min_value=10.0, max_value=150.0, value=60.0)
    ldl = st.number_input("LDL (Bad Cholesterol)", min_value=20.0, max_value=300.0, value=100.0)
    crp_level = st.number_input("CRP Level", min_value=0.0, max_value=100.0, value=5.0)
    homocysteine_level = st.number_input("Homocysteine Level", min_value=0.0, max_value=50.0, value=10.0)

with col3:
    st.subheader("Lifestyle & History")
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol_intake = st.number_input("Alcohol Intake", min_value=0.0, max_value=100.0, value=0.0)
    salt_intake = st.number_input("Salt Intake", min_value=0.0, max_value=50.0, value=5.0)
    
    physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
    sugar_consumption = st.selectbox("Sugar Consumption", ["Low", "Moderate", "High"])
    
    family_history = st.selectbox("Family History of Disease?", ["No", "Yes"])
    low_hdl_cholesterol = st.selectbox("Clinically Low HDL?", ["No", "Yes"])
    high_ldl_cholesterol = st.selectbox("Clinically High LDL?", ["No", "Yes"])
    
    education_level = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"])

# --- Prediction Logic ---
st.markdown("---")
if st.button("Predict & Route Patient (تشخيص وتوجيه المريض)", use_container_width=True):
    
    # 1. Data Mapping & Encoding (Matching Phase 1 Pipeline)
    gender_mapped = 1 if gender == "Male" else 0
    binary_map = {"No": 0, "Yes": 1}
    ordinal_map = {"Low": 0, "Moderate": 1, "High": 2}
    edu_map = {"Primary": 0, "Secondary": 1, "Tertiary": 2}
    
    # One-Hot Encoding manual resolution
    smoking_Former = 1 if smoking == "Former" else 0
    smoking_Never = 1 if smoking == "Never" else 0
    
    employment_status_Retired = 1 if employment_status == "Retired" else 0
    employment_status_Unemployed = 1 if employment_status == "Unemployed" else 0

    # 2. Build the DataFrame exact feature order
    input_data = {
        'gender': [gender_mapped],
        'age': [age],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'glucose': [glucose],
        'cholesterol': [cholesterol],
        'sleep_hours': [sleep_hours],
        'triglycerides': [150.0], # Default standard
        'physical_activity': [ordinal_map[physical_activity]],
        'family_history': [binary_map[family_history]],
        'stress_level': [ordinal_map[stress_level]],
        'low_hdl_cholesterol': [binary_map[low_hdl_cholesterol]],
        'high_ldl_cholesterol': [binary_map[high_ldl_cholesterol]],
        'sugar_consumption': [ordinal_map[sugar_consumption]],
        'crp_level': [crp_level],
        'homocysteine_level': [homocysteine_level],
        'systolic_bp': [systolic_bp],
        'diastolic_bp': [diastolic_bp],
        'alcohol_intake': [alcohol_intake],
        'salt_intake': [salt_intake],
        'heart_rate': [heart_rate],
        'hdl': [hdl],
        'ldl': [ldl],
        'education_level': [edu_map[education_level]],
        'smoking_Former': [smoking_Former],
        'smoking_Never': [smoking_Never],
        'employment_status_Retired': [employment_status_Retired],
        'employment_status_Unemployed': [employment_status_Unemployed]
    }
    
    df_patient = pd.DataFrame(input_data)
    
    # 3. Make Predictions
    st.subheader("نتائج التشخيص (Diagnostic Results)")
    targets = ['heart_disease', 'hypertension', 'diabetes']
    
    predictions = {}
    for target in targets:
        model = models_dict[target]
        # Predict probability of class 1 (Positive)
        prob = model.predict_proba(df_patient)[0][1] 
        predictions[target] = prob
        
    # 4. Display Results and Routing
    c1, c2, c3 = st.columns(3)
    
    # Heart Disease
    with c1:
        if predictions['heart_disease'] >= 0.5:
            st.error(f"⚠️ Heart Disease Detected\nProbability: {predictions['heart_disease']:.1%}")
            st.info("🩺 Route to: **Cardiology (طبيب قلب)**")
        else:
            st.success(f"✅ No Heart Disease\nProbability: {predictions['heart_disease']:.1%}")

    # Hypertension
    with c2:
        if predictions['hypertension'] >= 0.5:
            st.error(f"⚠️ Hypertension Detected\nProbability: {predictions['hypertension']:.1%}")
            st.info("🩺 Route to: **Internal Medicine / Cardiology (باطنة / قلب)**")
        else:
            st.success(f"✅ No Hypertension\nProbability: {predictions['hypertension']:.1%}")

    # Diabetes
    with c3:
        if predictions['diabetes'] >= 0.5:
            st.error(f"⚠️ Diabetes Detected\nProbability: {predictions['diabetes']:.1%}")
            st.info("🩺 Route to: **Endocrinology (طبيب غدد صماء وسكر)**")
        else:
            st.success(f"✅ No Diabetes\nProbability: {predictions['diabetes']:.1%}")
            
    if all(p < 0.5 for p in predictions.values()):
        st.balloons()
        st.success("🎉 المريض لا يعاني من أي من هذه الأمراض الثلاثة. (Patient is healthy regarding these conditions)")
