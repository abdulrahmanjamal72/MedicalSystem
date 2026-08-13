# Smart Medical Diagnosis System

This repository contains an end-to-end machine learning framework that predicts a patient's likelihood of diabetes, hypertension, and heart disease from initial intake data, and routes them to the appropriate specialist. Developed as the final Capstone Project for the NTI Machine Learning Training program.

The system combines a tuned multi-output Random Forest baseline with the TabPFN tabular foundation model inside a stacking ensemble. It is served through a local Streamlit application for interactive, real-time predictions.

---

## Disclaimer

**This system is a decision-support and routing aid, not a diagnostic device.** It has not undergone any clinical or regulatory validation. Performance metrics evaluated on this dataset are not evidence of real-world diagnostic accuracy.

---

## Table of Contents

1. [Problem Statement & Motivation]
2. [Dataset & Data Integrity]
3. [System Architecture]
4. [Methodology & Performance]
5. [Installation & Usage]
6. [Future Work]
7. [Team Members]

---

## Problem Statement & Motivation

Global healthcare systems face a significant and well-documented triage inefficiency.

* Mis-triage error rates in emergency settings are reported between 15% and 33%.
* In the United States alone, at least 12 million individuals suffer from diagnostic errors annually, with misdiagnosis accounting for 5-17% of adverse hospital events.

**Objective:** Build a machine learning system that, given a patient's intake vitals and clinical indicators, predicts the probability of diabetes, hypertension, and heart disease, and uses those predictions to recommend a specialist route (e.g., Endocrinology, Cardiology), packaged behind a simple local web interface.

---

## Dataset & Data Integrity

The project uses the "Data Warehouse Multiclass" dataset, comprising over 280,000 patient records across 39 attributes, including demographic information, vital signs, lab values, and lifestyle indicators.

To ensure absolute methodological rigor:

* **Leakage Mitigation:** Five leakage columns derived directly from the targets were removed.
* **Deduplication:** Over 2,200 duplicate records were eliminated to prevent the model from memorizing repeated patients.
* **Encoding:** We applied binary encoding for Yes/No fields, ordinal encoding for ranked levels, and one-hot encoding (with `drop_first=True`) for nominal variables to avoid the dummy-variable trap.

---

## System Architecture

The production pipeline follows a simple, three-stage architecture: **intake → inference → routing**.

```text
Patient Intake Form (Streamlit UI)
                  │
                  ▼
Preprocessing (encoding, scaling to match training pipeline)
                  │
                  ▼
Base Models: Random Forest + TabPFN
                  │
                  ▼
Meta-Model (Logistic Regression) -> calibrated probability per disease
                  │
                  ▼
Thresholding -> Specialist Recommendation (Cardiology / Endocrinology / ...)

```

*(Architecture flow represented above)*

---

## Methodology & Performance

The modeling approach progressed in three phases:

1. **Phase 1 - Baseline:** A MultiOutputClassifier wrapping a class-weighted RandomForestClassifier was tuned via GridSearchCV over tree depth and estimator count, scored with `f1_macro`.
2. **Phase 2 - Advanced Ensemble:** Out-of-fold (OOF) predictions were generated for both the Random Forest and TabPFN (a pretrained tabular foundation model) using 5-fold Stratified K-Fold cross-validation. A Logistic Regression meta-model was trained on these OOF probabilities.
3. **Phase 3 - Critical Self-Audit:** Because the final ROC-AUC scores approached 1.0, the team ran a dedicated investigation. A shallow depth-3 decision tree achieved only 0.81 AUC, ruling out simple single-feature leakage. The near-perfect scores are attributable to subtle, dataset-level statistical artifacts (the dataset was likely assembled from multiple distinct cohorts with synthetic imputation).

**Evaluation Metrics:**

* **Optimization:** Precision-Recall curves were used to derive a per-disease decision threshold that maximizes F1.
* **Final Ensemble Score:** ROC-AUC approaching 1.0 across all three targets (evaluated on a strictly unseen, duplicate-free test set).

---

## Installation & Usage

The application runs entirely locally with no external API calls.

**1. Clone the repository**

```bash
git clone https://github.com/abdulrahmanjamal72/Smart-Medical-Diagnosis-System.git
cd Smart-Medical-Diagnosis-System

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Launch the application**

```bash
streamlit run app.py

```

---

## Future Work

* Validate the full pipeline on an independent, prospectively collected, non-synthetic clinical dataset.
* Integrate with hospital Electronic Health Record (EHR) systems via API for direct intake.
* Extend deployment from a local Streamlit demo to a containerized, cloud-hosted service.
* Add model explainability (e.g., SHAP values) to the interface.

---

## Team Members

This project was developed by a team of four specializing across the pipeline:

| Team Member | Primary Responsibility |
| --- | --- |
| **Abdulrahman Jamal** | Data Engineering & Exploratory Data Analysis |
| **Marwan Ayman** | Machine Learning Modeling & Algorithm Design |
| **Hassan Raafat** | Pretrained Model and Out-Of Fold Predictions |
| **Abdelbaset Haitham** | Ensemble Learning (Stacking) and Deployment |


