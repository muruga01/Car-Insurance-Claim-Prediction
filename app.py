import streamlit as st
import pandas as pd
import joblib
import io
import re

# Load model and preprocessor
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('final_xgb_model.pkl')

# Function to parse max_torque and max_power if they are strings (e.g., "150 Nm@4500rpm")
def parse_torque_power(value, unit):
    if isinstance(value, str):
        match = re.match(r'(\d+\.?\d*)', value)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Invalid format for {unit}: {value}")
    return float(value)

# Streamlit App
st.title('Car Insurance Claim Predictor')

# Option to choose between single prediction or batch prediction
prediction_mode = st.radio("Select Prediction Mode:", ('Single Prediction', 'Batch Prediction (Upload Test File)'))

if prediction_mode == 'Single Prediction':
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
    max_torque_input = st.number_input('Max Torque (Nm)', min_value=0.0, value=150.0)
    max_power_input = st.number_input('Max Power (bhp)', min_value=0.0, value=100.0)

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
        'policy_tenure': policy_tenure,
        'age_of_car': age_of_car,
        'age_of_policyholder': age_of_policyholder,
        'population_density': population_density,
        'gross_weight': gross_weight,
        'ncap_rating': ncap_rating,
        'displacement': displacement,
        'cylinder': cylinder,
        'gear_box': gear_box,
        'turning_radius': turning_radius,
        'length': length,
        'width': width,
        'height': height,
        'max_torque': max_torque_input,  # Use correct column name
        'max_power': max_power_input,    # Use correct column name
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

else:
    # Batch Prediction
    st.header('Upload Test Dataset')
    uploaded_file = st.file_uploader("Choose test.csv", type=['csv'])
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            test_ids = test_df['policy_id']
            test_features = test_df.drop('policy_id', axis=1, errors='ignore')
            # Handle max_torque and max_power if they are strings
            if 'max_torque' in test_features.columns and test_features['max_torque'].dtype == 'object':
                test_features['max_torque'] = test_features['max_torque'].apply(lambda x: parse_torque_power(x, 'max_torque'))
            if 'max_power' in test_features.columns and test_features['max_power'].dtype == 'object':
                test_features['max_power'] = test_features['max_power'].apply(lambda x: parse_torque_power(x, 'max_power'))
            test_proc = preprocessor.transform(test_features)
            test_predictions = model.predict(test_proc)
            
            # Create submission file
            submission = pd.DataFrame({
                'policy_id': test_ids,
                'is_claim': test_predictions
            })
            
            # Match sample_submission format
            sample_submission = pd.read_csv('sample_submission.csv')
            submission = submission[sample_submission.columns]
            
            # Provide download button
            buffer = io.StringIO()
            submission.to_csv(buffer, index=False)
            st.download_button(
                label="Download Submission File",
                data=buffer.getvalue(),
                file_name='submission.csv',
                mime='text/csv'
            )
            st.write("Submission file generated successfully!")
        except Exception as e:
            st.error(f"Error processing test file: {str(e)}")

# Display EDA Visualizations
st.header('EDA Visualizations')
try:
    st.image('claim_distribution.png', caption='Claim Distribution')
    st.image('correlation_heatmap.png', caption='Correlation Heatmap')
    st.image('claim_by_fuel.png', caption='Claim Rate by Fuel Type')
    st.image('confusion_matrix.png', caption='Confusion Matrix')
except FileNotFoundError:
    st.warning("EDA images not found. Please run main.py first to generate visualizations.")