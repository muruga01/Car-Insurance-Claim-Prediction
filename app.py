import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('final_xgb_model.pkl')

# Streamlit App
st.title('Car Insurance Claim Predictor')

# Input fields for all features
st.header('Numerical Features')
policy_tenure = st.number_input('Policy Tenure (years)', min_value=0.0, max_value=10.0, value=1.0)
age_of_car = st.number_input('Age of Car (years)', min_value=0.0, max_value=20.0, value=5.0)
age_of_policyholder = st.number_input('Age of Policyholder', min_value=18, max_value=100, value=30)
population_density = st.number_input('Population Density', min_value=0, value=10000)
gross_weight = st.number_input('Gross Weight (kg)', min_value=0, value=1200)
ncap_rating = st.number_input('NCAP Rating', min_value=0, max_value=5, value=3)
displacement = st.number_input('Displacement (cc)', min_value=0, value=1200)
cylinder = st.number_input('Cylinders', min_value=1, max_value=12, value=4)
gear_box = st.number_input('Gear Box (speeds)', min_value=1, max_value=10, value=5)
turning_radius = st.number_input('Turning Radius (m)', min_value=0.0, value=5.0)
length = st.number_input('Vehicle Length (mm)', min_value=0, value=4000)
width = st.number_input('Vehicle Width (mm)', min_value=0, value=1800)
height = st.number_input('Vehicle Height (mm)', min_value=0, value=1500)
max_torque = st.number_input('Max Torque (Nm)', min_value=0.0, value=150.0)
max_power = st.number_input('Max Power (bhp)', min_value=0.0, value=100.0)

# Categorical features
st.header('Categorical Features')
area_cluster = st.selectbox('Area Cluster', options=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
make = st.selectbox('Make', options=[1, 2, 3, 4, 5])
segment = st.selectbox('Segment', options=['A', 'B1', 'B2', 'C1', 'C2', 'Utility'])
model = st.selectbox('Model', options=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11'])
fuel_type = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG'])
engine_type = st.selectbox('Engine Type', options=['1.2 Petrol', '1.5 Diesel', '1.0 Petrol', 'CNG'])
airbags = st.selectbox('Airbags', options=[1, 2, 4, 6])
is_esc = st.selectbox('ESC (Electronic Stability Control)', options=['Yes', 'No'])
is_adjustable_steering = st.selectbox('Adjustable Steering', options=['Yes', 'No'])
is_tpms = st.selectbox('TPMS (Tire Pressure Monitoring)', options=['Yes', 'No'])
is_parking_sensors = st.selectbox('Parking Sensors', options=['Yes', 'No'])
is_parking_camera = st.selectbox('Parking Camera', options=['Yes', 'No'])
rear_brakes_type = st.selectbox('Rear Brakes Type', options=['Drum', 'Disc'])
transmission_type = st.selectbox('Transmission Type', options=['Manual', 'Automatic'])
steering_type = st.selectbox('Steering Type', options=['Power', 'Electric', 'Manual'])
is_front_fog_lights = st.selectbox('Front Fog Lights', options=['Yes', 'No'])
is_rear_window_wiper = st.selectbox('Rear Window Wiper', options=['Yes', 'No'])
is_rear_window_washer = st.selectbox('Rear Window Washer', options=['Yes', 'No'])
is_rear_window_defogger = st.selectbox('Rear Window Defogger', options=['Yes', 'No'])
is_brake_assist = st.selectbox('Brake Assist', options=['Yes', 'No'])
is_power_door_locks = st.selectbox('Power Door Locks', options=['Yes', 'No'])
is_central_locking = st.selectbox('Central Locking', options=['Yes', 'No'])
is_power_steering = st.selectbox('Power Steering', options=['Yes', 'No'])
is_driver_seat_height_adjustable = st.selectbox('Driver Seat Height Adjustable', options=['Yes', 'No'])
is_day_night_rear_view_mirror = st.selectbox('Day/Night Rear View Mirror', options=['Yes', 'No'])
is_ecw = st.selectbox('ECW (Engine Check Warning)', options=['Yes', 'No'])
is_speed_alert = st.selectbox('Speed Alert', options=['Yes', 'No'])

# Collect inputs with correct column names
input_data = {
    'policy_tenure': float(policy_tenure),
    'age_of_car': float(age_of_car),
    'age_of_policyholder': float(age_of_policyholder),
    'population_density': float(population_density),
    'gross_weight': float(gross_weight),
    'ncap_rating': float(ncap_rating),
    'displacement': float(displacement),
    'cylinder': float(cylinder),
    'gear_box': float(gear_box),
    'turning_radius': float(turning_radius),
    'length': float(length),
    'width': float(width),
    'height': float(height),
    'max_torque': float(max_torque),
    'max_power': float(max_power),
    'area_cluster': area_cluster,
    'make': make,
    'segment': segment,
    'model': model,
    'fuel_type': fuel_type,
    'engine_type': engine_type,
    'airbags': airbags,
    'is_esc': is_esc,
    'is_adjustable_steering': is_adjustable_steering,
    'is_tpms': is_tpms,
    'is_parking_sensors': is_parking_sensors,
    'is_parking_camera': is_parking_camera,
    'rear_brakes_type': rear_brakes_type,
    'transmission_type': transmission_type,
    'steering_type': steering_type,
    'is_front_fog_lights': is_front_fog_lights,
    'is_rear_window_wiper': is_rear_window_wiper,
    'is_rear_window_washer': is_rear_window_washer,
    'is_rear_window_defogger': is_rear_window_defogger,
    'is_brake_assist': is_brake_assist,
    'is_power_door_locks': is_power_door_locks,
    'is_central_locking': is_central_locking,
    'is_power_steering': is_power_steering,
    'is_driver_seat_height_adjustable': is_driver_seat_height_adjustable,
    'is_day_night_rear_view_mirror': is_day_night_rear_view_mirror,
    'is_ecw': is_ecw,
    'is_speed_alert': is_speed_alert,
    'policy_id': 'ID_dummy'
}

# Single Prediction
if st.button('Predict'):
    try:
        input_df = pd.DataFrame([input_data])
        processed = preprocessor.transform(input_df)
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]
        st.write(f'**Claim Prediction**: {"Yes" if pred else "No"} (Probability: {prob:.2f})')
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Display EDA Visualizations
st.header('EDA Visualizations')
try:
    st.image('claim_distribution.png', caption='Claim Distribution')
    st.image('correlation_heatmap.png', caption='Correlation Heatmap')
    st.image('claim_by_fuel.png', caption='Claim Rate by Fuel Type')
    st.image('confusion_matrix.png', caption='Confusion Matrix')
except FileNotFoundError:
    st.warning("EDA images not found. Please run main.py first to generate visualizations.")