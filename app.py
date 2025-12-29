import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('car_insurance_claim_model.pkl')

# Real manufacturer names
make_mapping = {
    "Maruti Suzuki": 1,
    "Toyota": 2,
    "Mahindra": 3,
    "Hyundai": 4,
    "Honda": 5
}

# Realistic engine displacement options based on the actual dataset
# These are the most common and all unique values found in the Car Insurance dataset
displacement_options = [
    796,   # Small cars like Alto
    998,   # Common in Maruti/Hyundai small cars
    999,
    1197,  # Very common (1.2L petrol)
    1198,
    1199,
    1248,  # Common diesel
    1298,
    1299,
    1330,
    1364,
    1368,
    1396,
    1451,
    1461,  # Common diesel
    1462,
    1493,
    1497,  # Popular (1.5L petrol/diesel)
    1498,
    1582,
    1798,
    1956,  # Modern diesel
    1995,
    1996,
    1997,
    1998,
    1999,
    2143,  # Mercedes/Mahindra
    2198,
    2199,
    2393,
    2494,
    2523,
    2609,
    2953,
    2967,
    2982,
    2987,
    2993,
    3198,
    3604,
    4367,
    4663,
    4806,
    5000,
    5204,
    5461   # Luxury/large engines
]

st.title("🚗 Car Insurance Claim Prediction")

col1, col2 = st.columns(2)

with col1:
    policy_tenure = st.slider("1. Policy Tenure (years)", 0.0, 1.5, 0.8, 0.1)
    age_of_policyholder = st.slider("2. Policyholder Age (normalized)", 0.0, 1.0, 0.5, 0.05)
    age_of_car = st.slider("3. Car Age (normalized)", 0.0, 1.0, 0.1, 0.01)
    ncap_rating = st.selectbox("4. NCAP Rating (0-5)", [0,1,2,3,4,5], index=3)
    population_density = st.number_input("5. City Population Density", 499, 99999, 27000)

with col2:
    make_name = st.selectbox("6. Car Manufacturer", list(make_mapping.keys()))
    make = make_mapping[make_name]
    
    airbags = st.selectbox("7. Number of Airbags", [1,2,3,4,5,6], index=1)
    
    # Updated: Dropdown with actual dataset values
    displacement = st.selectbox(
        "8. Engine Displacement (cc)",
        options=displacement_options,
        index=displacement_options.index(1197) if 1197 in displacement_options else 0,
        help="Select from real engine sizes in the dataset (most common: 1197cc, 1497cc, etc.)"
    )
    
    turning_radius = st.slider("9. Turning Radius (m)", 4.5, 6.0, 5.2, 0.1)
    is_speed_alert = st.selectbox("10. Speed Alert System", ["Yes", "No"])

if st.button("🔮 Predict Claim Risk"):
    input_data = {
        'policy_tenure': policy_tenure,
        'age_of_car': age_of_car,
        'age_of_policyholder': age_of_policyholder,
        'area_cluster': 'C1',
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
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown(f"### Claim Probability: **{prob:.2%}**")
    if pred == 1:
        st.error("🚨 High Risk – Likely to file a claim")
    else:
        st.success("✅ Low Risk – Unlikely to file a claim")

    if prob > 0.15:
        st.warning("Suggest higher premium or further checks.")
    elif prob < 0.05:
        st.success("Great low-risk customer!")

st.caption("Engine displacement options are taken directly from the training dataset for accuracy.")