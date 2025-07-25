import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import shap
import matplotlib.pyplot as plt

# --- Constants ---
CLUSTER_LABELS = {
    0: "ChronicCare_Obese_Mixed",
    1: "RoutineCare_Stable", 
    2: "Diagnostics_Monitoring",
    3: "HighRisk_Hypertension",
    4: "UrgentCare_Diabetes"
}

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load pre-trained models with caching"""
    return {
        'scaler': pickle.load(open('scaler.sav', 'rb')),
        'model': pickle.load(open('svc_model.sav', 'rb')),
        'explainer': shap.TreeExplainer(pickle.load(open('svc_model.sav', 'rb')))
    }

# --- Prediction Logic ---
def predict(features):
    """Run prediction pipeline"""
    models = load_models()
    scaled_features = models['scaler'].transform([features])
    prediction = models['model'].predict(scaled_features)[0]
    return prediction, models['explainer'].shap_values(scaled_features)

# --- UI Components ---
def input_sidebar():
    """Render input controls"""
    st.sidebar.header("Patient Parameters")
    return [
        st.sidebar.number_input("Age", 0, 120, 45),
        st.sidebar.number_input("Hospital Stay (days)", 0, 365, 3),
        st.sidebar.number_input("Billing Amount ($)", 0.0, 100000.0, 2500.0),
        st.sidebar.selectbox("Admission Type", ["ICU", "ER", "OP"]),
        st.sidebar.radio("Test Results", ["Positive", "Negative"])
    ]

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Patient Segmentation AI",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("AI-Powered Patient Segmentation")
    st.image("healthcare_banner.jpg", use_column_width=True)
    
    # Input/Output
    inputs = input_sidebar()
    if st.sidebar.button("Predict"):
        cluster, shap_values = predict(inputs)
        
        # Results Display
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Predicted Cluster**: {CLUSTER_LABELS[cluster]}")
            st.dataframe(pd.DataFrame({
                'Feature': ['Age', 'Stay', 'Billing', 'Admission', 'Test'],
                'Value': inputs
            }))
            
        with col2:
            # SHAP Visualization
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, feature_names=['Age','Stay','Billing','Adm_ICU','Adm_ER','Adm_OP','Test_Pos'])
            st.pyplot(fig)

    # Model Info
    with st.expander("About This Project"):
        st.markdown("""
        - **Clustering**: K-Means identified 5 clinically distinct groups
        - **Classification**: SVM achieved 94% test accuracy
        - **Data**: Synthetic dataset simulating US hospital records
        """)

if __name__ == "__main__":
    main()