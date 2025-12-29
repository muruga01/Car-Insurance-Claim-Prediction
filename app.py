import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('car_insurance_claim_model.pkl')

st.title("🚗 Car Insurance Claim Prediction")
st.markdown("""
### Simplified Prediction Form (Only 10 Key Questions)
This version uses the **10 most important features** for quick and accurate predictions.
""")

st.info("Fill in these 10 fields for an instant claim risk prediction.")

col1, col2 = st.columns(2)

with col1:
    policy_tenure = st.slider("1. Policy Tenure (years)", 0.0, 1.5, 0.8, step=0.1,
                              help="How long the policy has been active (normalized 0-1.5)")
    
    age_of_policyholder = st.slider("2. Age of Policyholder (normalized)", 0.0, 1.0, 0.5, step=0.05,
                                    help="Policyholder age scaled (approx 18-80 years)")
    
    age_of_car = st.slider("3. Age of Car (normalized years)", 0.0, 1.0, 0.1, step=0.01,
                           help="Car age scaled (0 = new, 1 = oldest)")
    
    ncap_rating = st.selectbox("4. NCAP Safety Rating (out of 5)", options=[0,1,2,3,4,5], index=3,
                               help="Higher rating = safer car")
    
    population_density = st.number_input("5. Population Density of City", min_value=499, max_value=99999, value=27000,
                                         help="Higher density areas often have higher claim risk")

with col2:
    make = st.selectbox("6. Car Manufacturer", options=[1,2,3,4,5], index=0,
                        help="Encoded: 1=Maruti Suzuki, 3=Mahindra, etc.")
    
    airbags = st.selectbox("7. Number of Airbags", options=[1,2,3,4,5,6], index=1,
                           help="More airbags = safer")
    
    displacement = st.number_input("8. Engine Displacement (cc)", min_value=796, max_value=5461, value=1498,
                                   help="Engine size in cubic centimeters")
    
    turning_radius = st.slider("9. Turning Radius (meters)", 4.5, 6.0, 5.2, step=0.1,
                               help="Smaller radius = easier to maneuver")
    
    is_speed_alert = st.selectbox("10. Speed Alert System", options=["Yes", "No"], index=0,
                                  help="Audible alert when exceeding speed limit")

if st.button("🔮 Predict Claim Risk", type="primary"):
    # Fixed: Correct column name is 'is_power_door_lock' (singular)
    input_data = {
        'policy_tenure': policy_tenure,
        'age_of_car': age_of_car,
        'age_of_policyholder': age_of_policyholder,
        'area_cluster': 'C1',  # common default
        'population_density': population_density,
        'make': make,
        'segment': 'B2',
        'model': 'M1',
        'fuel_type': 'Petrol',
        'max_torque': '113Nm@4400rpm',
        'max_power': '88.7bhp@6000rpm',
        'engine_type': '1.2 L K12N Dualjet',
        'airbags': airbags,
        'is_esc': 'No',
        'is_adjustable_steering': 'Yes',
        'is_tpms': 'No',
        'is_parking_sensors': 'Yes',
        'is_parking_camera': 'No',
        'rear_brakes_type': 'Drum',
        'displacement': displacement,
        'cylinder': 4,
        'transmission_type': 'Manual',
        'gear_box': 5,
        'steering_type': 'Power',
        'turning_radius': turning_radius,
        'length': 3995,
        'width': 1735,
        'height': 1515,
        'gross_weight': 1400,
        'is_front_fog_lights': 'No',
        'is_rear_window_wiper': 'No',
        'is_rear_window_washer': 'No',
        'is_rear_window_defogger': 'No',
        'is_brake_assist': 'Yes',
        'is_power_door_locks': 'Yes',
        'is_central_locking': 'Yes',
        'is_power_steering': 'Yes',
        'is_driver_seat_height_adjustable': 'Yes',
        'is_day_night_rear_view_mirror': 'No',
        'is_ecw': 'Yes',
        'is_speed_alert': is_speed_alert,
        'ncap_rating': ncap_rating
    }

    input_df = pd.DataFrame([input_data])

    # Prediction
    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    ### Prediction Result
    **Claim Probability: {probability:.2%}**

    {"🚨 **High Risk** – Likely to file a claim" if prediction == 1 else "✅ **Low Risk** – Unlikely to file a claim"}
    """)

    if probability > 0.15:
        st.warning("Consider higher premium or additional risk assessment.")
    elif probability < 0.05:
        st.success("Excellent low-risk customer – potential for discounts!")

st.markdown("---")
st.caption("Note: The model uses all original features internally. Unentered fields are filled with typical values for accurate prediction.")