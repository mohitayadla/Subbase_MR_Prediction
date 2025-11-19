import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Resilient Modulus Prediction - Base",
    page_icon="🏗️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏗️ Resilient Modulus Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict resilient modulus for base layer using machine learning model</div>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        model = joblib.load('base_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'base_model.pkl' is in the same directory as this app.")
        return None

# Load the model
base_model = load_model()

st.markdown("---")

# Input Parameters Section
st.markdown("### Input Parameters for Base")

# Input fields
layer_thickness = st.number_input("Layer Thickness (mm)", min_value=0.0, value=150.0, step=10.0, help="Enter the thickness of the base layer")

no_4_passing = st.number_input("NO. 4 Passing (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1, help="Percentage passing through No. 4 sieve")

no_10_passing = st.number_input("NO. 10 Passing (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1, help="Percentage passing through No. 10 sieve")

no_40_passing = st.number_input("NO. 40 Passing (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1, help="Percentage passing through No. 40 sieve")

no_80_passing = st.number_input("NO. 80 Passing (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1, help="Percentage passing through No. 80 sieve")

no_200_passing = st.number_input("NO. 200 Passing (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, help="Percentage passing through No. 200 sieve")

liquid_limit = st.number_input("Liquid Limit (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Liquid limit of the soil")

plastic_limit = st.number_input("Plastic Limit (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1, help="Plastic limit of the soil")

plasticity_index = st.number_input("Plasticity Index", min_value=0.0, max_value=100.0, value=10.0, step=0.1, help="Plasticity index (LL - PL)")

# AASHTO Soil Classification Dropdown
aashto_options = ["A-1-a", "A-1-b", "A-2-4", "A-2-6", "A-2-7", "A-3", "A-4", "A-5", "A-6","A-7-6"]
aashto_class_text = st.selectbox(
    "AASHTO Soil Classification", 
    options=aashto_options,
    index=0,
    help="Select the AASHTO soil classification"
)

# Encode AASHTO class to number (0-11) based on your training
# Map each class to its encoded value
aashto_encoding = {
    "A-1-a": 0, "A-1-b": 1, "A-2-4": 2, 
    "A-2-6": 3, "A-2-7": 4, "A-3": 5, "A-4": 6, 
    "A-5": 7, "A-6": 8, "A-7-6": 9
}
aashto_class = aashto_encoding[aashto_class_text]

spec_gravity = st.number_input("Specific Gravity", min_value=2.0, max_value=3.5, value=2.65, step=0.01, help="Specific gravity of soil solids")

max_lab_dry_density = st.number_input("Maximum Lab Dry Density (g/cm³)", min_value=1.0, max_value=3.0, value=2.0, step=0.01, help="Maximum dry density from laboratory compaction test")

optimum_lab_moisture = st.number_input("Optimum Lab Moisture Content (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1, help="Optimum moisture content from laboratory test")

hydraulic_conductivity = st.number_input("Hydraulic Conductivity (cm/s)", min_value=0.0, value=1e-7, format="%.2e", help="Coefficient of permeability")

# Prepare input data
input_data = pd.DataFrame({
    'REPR_THICKNESS': [layer_thickness],
    'NO_4_PASSING': [no_4_passing],
    'NO_10_PASSING': [no_10_passing],
    'NO_40_PASSING': [no_40_passing],
    'NO_80_PASSING': [no_80_passing],
    'NO_200_PASSING': [no_200_passing],
    'LIQUID_LIMIT': [liquid_limit],
    'PLASTIC_LIMIT': [plastic_limit],
    'PLASTICITY_INDEX': [plasticity_index],
    'AASHTO_SOIL_CLASS': [aashto_class],
    'SPEC_GRAVITY': [spec_gravity],
    'MAX_LAB_DRY_DENSITY': [max_lab_dry_density],
    'OPTIMUM_LAB_MOISTURE': [optimum_lab_moisture],
    'HYDRAULIC_CONDUCTIVITY': [hydraulic_conductivity]
})

# Predict button - Only show if all inputs are filled
st.markdown("---")

# Check if all required inputs are provided (non-zero/non-default values that make sense)
all_inputs_filled = (
    layer_thickness > 0 and
    no_4_passing > 0 and
    no_10_passing > 0 and
    no_40_passing > 0 and
    no_80_passing > 0 and
    no_200_passing > 0 and
    liquid_limit > 0 and
    plastic_limit > 0 and
    plasticity_index >= 0 and
    spec_gravity > 0 and
    max_lab_dry_density > 0 and
    optimum_lab_moisture > 0 and
    hydraulic_conductivity > 0
)

if not all_inputs_filled:
    st.warning("⚠️ Please fill in all input parameters before predicting.")
    st.button("🔮 Predict Resilient Modulus", type="primary", use_container_width=True, disabled=True)
else:
    if st.button("🔮 Predict Resilient Modulus", type="primary", use_container_width=True):
        if base_model is not None:
            try:
                prediction = base_model.predict(input_data)[0]
                
                # Display result in a nice box
                st.balloons()
                st.success(f"### ✅ Prediction Successful!")
                st.metric(label="Predicted Resilient Modulus", value=f"{prediction:.2f} MPa")
                
                # Optional: Show input summary
                with st.expander("📋 View Input Summary"):
                    st.dataframe(input_data.T, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.warning("Please check if your model expects these exact feature names and order.")
        else:
            st.error("❌ Model not loaded. Please check that 'base_model.pkl' exists in the app directory.")

# Footer
st.markdown("---")
st.markdown("**Note:** Ensure all input values are accurate for reliable predictions. The model uses machine learning to estimate resilient modulus based on material properties.")