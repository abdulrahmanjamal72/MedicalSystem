# AI Medical Symptom Routing System

## About The Project
This project is an AI-driven healthcare recommendation system developed as our final project for the **NTI Machine Learning Training**. The system utilizes multiple machine learning models to predict potential diseases based on user-inputted symptoms or medical data, and subsequently routes the patient to the most appropriate medical specialist.

## Key Features
- **Symptom Analysis:** Accepts medical symptoms or test readings as input.
- **Multi-Model Disease Prediction:** Utilizes distinct ML models tailored for various diseases (e.g., Heart Disease, Diabetes).
- **Specialist Routing:** Recommends the exact medical specialty the user should visit based on the prediction.
- **User-Friendly Interface:** A simple web interface for seamless interaction.

## Team & Roles (Work Distribution)
This project is collaboratively built by a team of 4 members, with equally distributed responsibilities:

| Team Member | Key Responsibilities |
| :--- | :--- |
| **Abdulrahman Jamal** | Data gathering, cleaning (handling missing values), EDA, and feature engineering to prepare the CSV datasets. |
| **Marwan Ayman** | Building, training, and tuning the multiple machine learning models (Hyperparameter tuning, evaluating accuracy). |
| **Hassan Raafat** | Integrating the models and deploying them using a web framework (e.g., Streamlit) to create the routing system. |
| **Abdelbaset Haitham** | System verification, integration testing (ensuring seamless data flow from UI to models), and presentation slides. |

## Technologies Used
* **Data Processing:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Deployment:** Streamlit
* **Version Control:** GitHub

## Dataset
The project utilizes the **MIMIC-IV-ED** dataset from PhysioNet, representing real-world Emergency Department data. It includes vital signs, triage information, and discharge diagnoses. The data was preprocessed to map final diagnoses (ICD Codes) to specific medical specialties for accurate patient routing.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/abdulrahmanjamal72/Smart-Medical-Diagnosis-System.git](https://github.com/abdulrahmanjamal72/Smart-Medical-Diagnosis-System.git)
