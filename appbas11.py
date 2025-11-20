import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# Page configuration
st.set_page_config(page_title="Resilient Modulus Prediction", page_icon="🏗️", layout="wide")

# Title and description
st.title("🏗️ Resilient Modulus Prediction System")
st.markdown("Predict resilient modulus for base layer using machine learning model")

st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Material Properties")
    
    repr_thickness = st.number_input(
        "Layer Thickness (mm)",
        min_value=0.0,
        value=150.0,
        step=1.0,
        help="Representative thickness of the base layer"
    )
    
    no_4_passing = st.number_input(
        "NO. 4 Passing (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.1,
        help="Percentage passing through No. 4 sieve"
    )
    
    no_10_passing = st.number_input(
        "NO. 10 Passing (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=0.1,
        help="Percentage passing through No. 10 sieve"
    )
    
    no_40_passing = st.number_input(
        "NO. 40 Passing (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=0.1,
        help="Percentage passing through No. 40 sieve"
    )
    
    no_80_passing = st.number_input(
        "NO. 80 Passing (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1,
        help="Percentage passing through No. 80 sieve"
    )
    
    no_200_passing = st.number_input(
        "NO. 200 Passing (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=0.1,
        help="Percentage passing through No. 200 sieve"
    )

with col2:
    st.subheader("Soil Characteristics")
    
    liquid_limit = st.number_input(
        "Liquid Limit (%)",
        min_value=0.0,
        value=25.0,
        step=0.1,
        help="Liquid limit of the soil"
    )
    
    plastic_limit = st.number_input(
        "Plastic Limit (%)",
        min_value=0.0,
        value=15.0,
        step=0.1,
        help="Plastic limit of the soil"
    )
    
    plasticity_index = st.number_input(
        "Plasticity Index",
        min_value=0.0,
        value=10.0,
        step=0.1,
        help="Plasticity index (LL - PL)"
    )
    
    aashto_soil_class = st.selectbox(
        "AASHTO Soil Class",
        options=["A-1-a", "A-1-b", "A-2-4", "A-2-5", "A-2-6", "A-2-7", "A-3", "A-4", "A-5", "A-6", "A-7-5", "A-7-6"],
        help="AASHTO soil classification"
    )
    
    spec_gravity = st.number_input(
        "Specific Gravity",
        min_value=0.0,
        value=2.65,
        step=0.01,
        help="Specific gravity of soil particles"
    )
    
    max_lab_dry_density = st.number_input(
        "Maximum Lab Dry Density (g/cm³)",
        min_value=0.0,
        value=2.0,
        step=0.01,
        help="Maximum dry density from laboratory tests"
    )
    
    optimum_lab_moisture = st.number_input(
        "Optimum Lab Moisture Content (%)",
        min_value=0.0,
        value=10.0,
        step=0.1,
        help="Optimum moisture content"
    )
    
    hydraulic_conductivity = st.number_input(
        "Hydraulic Conductivity (cm/s)",
        min_value=0.0,
        value=1.0e-7,
        format="%.2e",
        help="Hydraulic conductivity of the material"
    )

st.markdown("---")

# Predict button
if st.button("🔍 Predict Resilient Modulus", type="primary", use_container_width=True):
    # Validate all inputs are provided
    if all([
        repr_thickness, no_4_passing, no_10_passing, no_40_passing,
        no_80_passing, no_200_passing, liquid_limit, plastic_limit,
        plasticity_index, aashto_soil_class, spec_gravity,
        max_lab_dry_density, optimum_lab_moisture, hydraulic_conductivity
    ]):
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'REPR_THICKNESS': [repr_thickness],
            'NO_4_PASSING': [no_4_passing],
            'NO_10_PASSING': [no_10_passing],
            'NO_40_PASSING': [no_40_passing],
            'NO_80_PASSING': [no_80_passing],
            'NO_200_PASSING': [no_200_passing],
            'LIQUID_LIMIT': [liquid_limit],
            'PLASTIC_LIMIT': [plastic_limit],
            'PLASTICITY_INDEX': [plasticity_index],
            'AASHTO_SOIL_CLASS': [aashto_soil_class],
            'SPEC_GRAVITY': [spec_gravity],
            'MAX_LAB_DRY_DENSITY': [max_lab_dry_density],
            'OPTIMUM_LAB_MOISTURE': [optimum_lab_moisture],
            'HYDRAULIC_CONDUCTIVITY': [hydraulic_conductivity]
        })
        
        # Display input summary
        st.success("✅ All required parameters provided!")
        
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_data.T, use_container_width=True)
        
        # Placeholder for model prediction
        st.info("⚠️ **Note:** To get actual predictions, you need to load your trained model. Replace the demo calculation below with your actual model.")
        
        # Demo calculation (replace with actual model prediction)
        # Example: predicted_modulus = model.predict(input_data)[0]
        
        # Simple demo formula (NOT a real prediction - replace with your model)
        predicted_modulus = (
            repr_thickness * 0.5 + 
            max_lab_dry_density * 1000 + 
            spec_gravity * 500 - 
            optimum_lab_moisture * 50 +
            no_4_passing * 2
        )
        
        # Display prediction
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                label="Predicted Resilient Modulus",
                value=f"{predicted_modulus:.2f} MPa"
            )
        
        with result_col2:
            st.metric(
                label="Confidence Level",
                value="Demo Mode"
            )
        
        with result_col3:
            st.metric(
                label="Model Status",
                value="Replace with Real Model"
            )
        
        st.warning("🔧 **Implementation Required:** Load your trained model using `pickle` or `joblib` and replace the demo calculation with `model.predict(input_data)`")
        
        # Code snippet for model loading
        with st.expander("💡 How to integrate your model"):
            st.code("""
# Add this at the top of your script:
import pickle

# Load your trained model
@st.cache_resource
def load_model():
    with open('resilient_modulus_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Then replace the demo calculation with:
predicted_modulus = model.predict(input_data)[0]
            """, language="python")
    
    else:
        st.error("❌ Please fill in all required parameters before making a prediction!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Resilient Modulus Prediction System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)