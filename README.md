Patient Intake Form (Streamlit UI)
                  │
                  ▼
Preprocessing (Encoding, Scaling to match training pipeline)
                  │
                  ▼
Base Models: Multi-Output Random Forest + TabPFN
                  │
                  ▼
Meta-Model (Logistic Regression) ──► Calibrated Probability per Disease
                  │
                  ▼
Thresholding ──► Specialist Recommendation (Cardiology / Endocrinology / etc.)
```[cite: 1]

---

## 📊 Dataset & Preprocessing Pipeline
* **Source Data:** "Data Warehouse Multiclass" dataset comprising over 280,000 patient records across 39 attributes[cite: 1].
* **Data Integrity:** Removed 5 direct target-leakage columns and eliminated over 2,200 duplicate records[cite: 1].
* **Feature Engineering & Encoding:**
  * Binary encoding for Yes/No status fields[cite: 1].
  * Ordinal encoding for ranked levels (Low/Moderate/High → 0/1/2)[cite: 1].
  * One-hot encoding (`drop_first=True`) for nominal categorical variables[cite: 1].
  * Feature redundancy pruning via Pearson correlation and verification using Spearman correlation[cite: 1].

---

## 🔬 Methodology & Modeling Strategy
1. **Baseline Model:** Multi-Output Random Forest Classifier tuned via `GridSearchCV` and evaluated using `f1_macro`[cite: 1].
2. **Advanced Stacking Ensemble:**
   * Out-Of-Fold (OOF) prediction generation using 5-Fold Stratified K-Fold cross-validation for both base learners (Random Forest + TabPFN)[cite: 1].
   * Meta-feature concatenation and meta-learner training using Logistic Regression[cite: 1].
3. **Decision Threshold Optimization:** Disease-specific decision thresholds derived via Precision-Recall curves to optimize F1 scores[cite: 1].

---

## 📈 Model Evaluation & Critical Audit

| Model | Key Metric | Result / Finding |
|---|---|---|
| Baseline Multi-Output RF | `f1_macro` | Benchmark for ensemble comparison[cite: 1] |
| Stacking Ensemble (RF + TabPFN) | ROC-AUC | Approaching 1.0 across all three targets[cite: 1] |
| Depth-3 Decision Tree (Control) | ROC-AUC | 0.81 (significantly lower than full ensemble)[cite: 1] |

### 🔍 Critical Self-Audit Summary
* **Investigation:** To audit the near-perfect ROC-AUC metrics, feature ablation experiments and model-complexity tests were conducted[cite: 1].
* **Findings:** Shallow control trees performed significantly lower (0.81 AUC), ruling out single-feature direct leakage[cite: 1]. The near-perfect scores stem from underlying statistical artifacts across multiple cohorts within the source dataset rather than true clinical separability[cite: 1].
* **Conclusion:** Complex models detect statistical fingerprints within the dataset, highlighting the necessity of validation on prospectively collected real-world clinical data[cite: 1].

---

## 🛠 Tech Stack
* **Language:** Python 3.x[cite: 1]
* **Data Manipulation:** Pandas, NumPy[cite: 1]
* **Machine Learning:** Scikit-Learn (RandomForestClassifier, MultiOutputClassifier, LogisticRegression), TabPFN[cite: 1]
* **Validation Strategy:** Stratified K-Fold CV, Out-Of-Fold (OOF) Predictions[cite: 1]
* **Serialization:** joblib[cite: 1]
* **Deployment UI:** Streamlit[cite: 1]
* **Version Control:** Git / GitHub[cite: 1]

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdulrahmanjamal72/Smart-Medical-Diagnosis-System.git
cd Smart-Medical-Diagnosis-System
```[cite: 1]

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```[cite: 1]

### 3. Launch the Web Application
```bash
streamlit run app.py
```[cite: 1]

---

## 👥 Team Members & Roles

| Team Member | Primary Responsibility |
|---|---|
| **Abdulrahman Jamal** | Data Engineering & Exploratory Data Analysis[cite: 1] |
| **Marwan Ayman** | Machine Learning Modeling & Algorithm Design[cite: 1] |
| **Hassan Raafat** | Pretrained Model & Out-Of-Fold Predictions[cite: 1] |
| **Abdelbaset Haitham** | Ensemble Learning (Stacking) & Deployment[cite: 1] |

---

## 🔮 Future Work
* Validate the complete pipeline on independent, prospectively collected clinical datasets[cite: 1].
* Integrate with hospital Electronic Health Record (EHR) systems via API for automated intake[cite: 1].
* Extend deployment to containerized, cloud-hosted services with authentication and audit logging[cite: 1].
* Incorporate model explainability methods (e.g., SHAP values) into the clinician interface[cite: 1].

---

## ⚠️ Disclaimer
This system is developed strictly as an academic decision-support and symptom-routing project[cite: 1]. It is **not** a certified diagnostic device and has not undergone clinical or regulatory validation[cite: 1].
