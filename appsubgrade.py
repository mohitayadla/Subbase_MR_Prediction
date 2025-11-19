import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Resilient Modulus Prediction - Subgrade",
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
st.markdown('<div class="sub-header">Predict resilient modulus for subgrade layer using machine learning model</div>', unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        model = joblib.load('subgrade_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'subgrade_model.pkl' is in the same directory as this app.")
        return None

# Load the model
subgrade_model = load_model()

st.markdown("---")

# Input Parameters Section
st.markdown("### Input Parameters for Subgrade")

# Input fields
no_4_passing = st.number_input("NO. 4 Passing (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1, help="Percentage passing through No. 4 sieve")

no_10_passing = st.number_input("NO. 10 Passing (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1, help="Percentage passing through No. 10 sieve")

no_40_passing = st.number_input("NO. 40 Passing (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1, help="Percentage passing through No. 40 sieve")

no_80_passing = st.number_input("NO. 80 Passing (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1, help="Percentage passing through No. 80 sieve")

no_200_passing = st.number_input("NO. 200 Passing (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, help="Percentage passing through No. 200 sieve")

coarse_sand = st.number_input("Coarse Sand (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Coarse sand content")

fine_sand = st.number_input("Fine Sand (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Fine sand content")

silt = st.number_input("Silt (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1, help="Silt content")

clay = st.number_input("Clay (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1, help="Clay content")

max_lab_dry_density = st.number_input("Maximum Lab Dry Density (g/cm³)", min_value=1.0, max_value=3.0, value=2.0, step=0.01, help="Maximum dry density from laboratory compaction test")

optimum_lab_moisture = st.number_input("Optimum Lab Moisture Content (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1, help="Optimum moisture content from laboratory test")

liquid_limit = st.number_input("Liquid Limit (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Liquid limit of the soil")

plastic_limit = st.number_input("Plastic Limit (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1, help="Plastic limit of the soil")

# Prepare input data with exact column names from your image
input_data = pd.DataFrame({
    'NO_4_PASSING': [no_4_passing],
    'NO_10_PASSING': [no_10_passing],
    'NO_40_PASSING': [no_40_passing],
    'NO_80_PASSING': [no_80_passing],
    'NO_200_PASSING': [no_200_passing],
    'COARSE_SAND': [coarse_sand],
    'FINE_SAND': [fine_sand],
    'SILT': [silt],
    'CLAY': [clay],
    'MAX_LAB_DRY_DENSITY': [max_lab_dry_density],
    'OPTIMUM_LAB_MOISTURE': [optimum_lab_moisture],
    'LIQUID_LIMIT': [liquid_limit],
    'PLASTIC_LIMIT': [plastic_limit]
})

# Predict button - Only show if all inputs are filled
st.markdown("---")

# Check if all required inputs are provided
all_inputs_filled = (
    no_4_passing > 0 and
    no_10_passing > 0 and
    no_40_passing > 0 and
    no_80_passing > 0 and
    no_200_passing > 0 and
    coarse_sand >= 0 and
    fine_sand >= 0 and
    silt >= 0 and
    clay >= 0 and
    max_lab_dry_density > 0 and
    optimum_lab_moisture > 0 and
    liquid_limit > 0 and
    plastic_limit > 0
)

if not all_inputs_filled:
    st.warning("⚠️ Please fill in all input parameters before predicting.")
    st.button("🔮 Predict Resilient Modulus", type="primary", use_container_width=True, disabled=True)
else:
    if st.button("🔮 Predict Resilient Modulus", type="primary", use_container_width=True):
        if subgrade_model is not None:
            try:
                prediction = subgrade_model.predict(input_data)[0]
                
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
            st.error("❌ Model not loaded. Please check that 'subgrade_model.pkl' exists in the app directory.")

# Footer
st.markdown("---")
st.markdown("**Note:** Ensure all input values are accurate for reliable predictions. The model uses machine learning to estimate resilient modulus based on material properties.")