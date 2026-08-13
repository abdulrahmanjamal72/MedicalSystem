import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# --- إعدادات تحميل النماذج من Hugging Face 
REPO_ID = "abdulrahmanjamal2/MedicalSystem" 
ARTIFACTS = ['our_models.pkl', 'tabpfn_models.pkl', 'meta_models.pkl', 'thresholds.pkl']

@st.cache_resource
def load_artifacts():
    artifacts_path = {}
    for artifact in ARTIFACTS:
        file_path = hf_hub_download(repo_id=REPO_ID, filename=artifact)
        artifacts_path[artifact] = joblib.load(file_path)
    return artifacts_path

st.title("AI Medical Symptom Routing Dashboard")
st.write("تطبيق ذكاء اصطناعي لتوجيه وتشخيص الأعراض الطبية.")

try:
    models = load_artifacts()
    st.success("تم تحميل النماذج بنجاح من Hugging Face! 🚀")
except Exception as e:
    st.error(f"خطأ في التحميل: {e}")
    st.stop()

st.header("بيانات المريض")
hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.0)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Predict"):
    st.success("جاري التنبؤ بنجاح!")
