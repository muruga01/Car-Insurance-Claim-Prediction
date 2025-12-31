import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import re
import os
from urllib.request import urlretrieve

# Page config
st.set_page_config(page_title="Car Insurance Claim Analytics Dashboard", layout="wide")

MODEL_URL = "https://drive.google.com/uc?export=download&id=1OnXQbocKohrJDl-R1YantsyWQZy6dy1I"
MODEL_PATH = "car_insurance_lgb_tuned_final.pkl"

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

@st.cache_resource(show_spinner="Downloading model (~300MB, first time only)... This may take 1-2 minutes.")
def load_model():
    if not os.path.exists(MODEL_PATH):
        urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# Manufacturer mapping
make_mapping = {1: "Maruti Suzuki", 2: "Toyota", 3: "Mahindra", 4: "Hyundai", 5: "Honda"}
df['make_name'] = df['make'].map(make_mapping)

# Calculate key metrics
total_policies = len(df)
claim_rate = df['is_claim'].mean()
high_risk_rate = (model.predict_proba(df.drop(['policy_id', 'is_claim'], axis=1))[:, 1] > 0.15).mean()

# ==================== Clean Feature Names Function ====================
def clean_feature_name(name):
    # Split on '__' (from ColumnTransformer)
    if '__' in name:
        transformer, feature = name.split('__', 1)
        name = feature
    
    # Replace underscores with spaces and capitalize
    name = name.replace('_', ' ').strip()
    
    # Special readable mappings for common features
    readable_map = {
        'policy tenure': 'Policy Tenure',
        'age of policyholder': 'Policyholder Age',
        'age of car': 'Car Age',
        'population density': 'City Population Density',
        'ncap rating': 'NCAP Safety Rating',
        'turning radius': 'Turning Radius (m)',
        'displacement': 'Engine Displacement (cc)',
        'airbags': 'Number of Airbags',
        'make': 'Car Manufacturer',
        'is speed alert': 'Speed Alert System',
        'is parking sensors': 'Parking Sensors',
        'is parking camera': 'Parking Camera',
        'is esc': 'Electronic Stability Control',
        'is tpms': 'Tyre Pressure Monitoring',
        'rear brakes type': 'Rear Brakes Type',
        'fuel type': 'Fuel Type',
        'transmission type': 'Transmission Type',
        'segment': 'Car Segment',
        'area cluster': 'Area Cluster'
    }
    
    return readable_map.get(name.lower(), name.title())

# # === Sidebar Tab Selector (Controls active tab reliably) ===
# st.sidebar.header("Navigation")
# tab_selection = st.sidebar.radio(
#     "Go to",
#     ["📊 Overview", "🔍 Data Exploration", "🤖 Model Insights", "🔮 Predict Claim"],
#     index=["📊 Overview", "🔍 Data Exploration", "🤖 Model Insights", "🔮 Predict Claim"].index(st.session_state.get('active_tab', "📊 Overview"))
# )

# Update session state
st.session_state.active_tab = tab_selection

# Create tabs (order must match radio options)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Data Exploration", "🤖 Model Insights", "🔮 Predict Claim"])

# ==================== TAB 1: Overview ====================
with tab1:
    if tab_selection == "📊 Overview":
        st.header("Insurance Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Policies", f"{total_policies:,}")
        col2.metric("Overall Claim Rate", f"{claim_rate:.2%}")
        col3.metric("High-Risk Policies", f"{high_risk_rate:.2%}")
        col4.metric("Average Policy Tenure", f"{df['policy_tenure'].mean():.2f} years")
        
        st.markdown("### Claim Distribution")
        fig_pie = px.pie(values=df['is_claim'].value_counts(), 
                         names=['No Claim', 'Claim'],
                         color_discrete_sequence=['#636EFA', '#EF553B'],
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==================== TAB 2: Data Exploration ====================
with tab2:
    if tab_selection == "🔍 Data Exploration":
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Claim Rate by Manufacturer")
            claim_by_make = df.groupby('make_name')['is_claim'].mean().sort_values(ascending=False).reset_index()
            fig_make = px.bar(claim_by_make, x='make_name', y='is_claim',
                              labels={'is_claim': 'Claim Rate', 'make_name': 'Manufacturer'},
                              color='is_claim', color_continuous_scale='Reds')
            st.plotly_chart(fig_make, use_container_width=True)
        
        with col2:
            st.subheader("Claim Rate by NCAP Safety Rating")
            claim_by_ncap = df.groupby('ncap_rating')['is_claim'].mean().reset_index()
            fig_ncap = px.bar(claim_by_ncap, x='ncap_rating', y='is_claim',
                              labels={'is_claim': 'Claim Rate', 'ncap_rating': 'NCAP Rating (0-5)'},
                              color='is_claim', color_continuous_scale='Blues')
            fig_ncap.update_xaxes(type='category')
            st.plotly_chart(fig_ncap, use_container_width=True)
        
        st.subheader("Claim Rate by Fuel Type")
        claim_by_fuel = df.groupby('fuel_type')['is_claim'].mean().sort_values(ascending=False).reset_index()
        fig_fuel = px.bar(claim_by_fuel, x='fuel_type', y='is_claim',
                          color='is_claim', color_continuous_scale='Greens')
        st.plotly_chart(fig_fuel, use_container_width=True)
        
        st.subheader("Policyholder Age vs Car Age (colored by Claim)")
        fig_scatter = px.scatter(df.sample(5000), x='age_of_policyholder', y='age_of_car',
                                 color='is_claim', opacity=0.6,
                                 labels={'age_of_policyholder': 'Policyholder Age (normalized)',
                                         'age_of_car': 'Car Age (normalized)'},
                                 color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==================== TAB 3: Model Insights ====================
with tab3:
    if tab_selection == "🤖 Model Insights":
        st.header("Model Performance & Explanations")
        
        feature_names_raw = model.named_steps['preprocessor'].get_feature_names_out()
        importances = model.named_steps['classifier'].feature_importances_
        
        clean_names = [clean_feature_name(f) for f in feature_names_raw]
        
        feat_df = pd.DataFrame({
            'feature': clean_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        st.subheader("Top 10 Most Predictive Features (LightGBM)")
        fig_imp = px.bar(feat_df, x='importance', y='feature', orientation='h',
                         labels={'importance': 'Feature Importance Score', 'feature': 'Feature'},
                         color='importance', color_continuous_scale='Viridis')
        fig_imp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

# ==================== TAB 4: Predict Claim ====================
with tab4:
    if tab_selection == "🔮 Predict Claim":
        st.header("Single Customer Claim Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            policy_tenure = st.slider("1. Policy Tenure (years)", 0.0, 1.5, 0.8, 0.1)
            age_of_policyholder = st.slider("2. Policyholder Age (normalized)", 0.0, 1.0, 0.5, 0.05)
            age_of_car = st.slider("3. Car Age (normalized)", 0.0, 1.0, 0.1, 0.01)
            ncap_rating = st.selectbox("4. NCAP Rating (0-5)", [0,1,2,3,4,5], index=3)
            population_density = st.number_input("5. City Population Density", 499, 99999, 27000)
        
        with col2:
            make_name = st.selectbox(
                "6. Car Manufacturer",
                options=["Maruti Suzuki", "Toyota", "Mahindra", "Hyundai", "Honda"]
            )
            make_mapping = {"Maruti Suzuki": 1, "Toyota": 2, "Mahindra": 3, "Hyundai": 4, "Honda": 5}
            make = make_mapping[make_name]
            
            airbags = st.selectbox("7. Number of Airbags", [1,2,3,4,5,6], index=1)
            
            displacement_options = sorted(df['displacement'].unique())
            displacement = st.selectbox(
                "8. Engine Displacement (cc)",
                options=displacement_options,
                index=displacement_options.index(1197) if 1197 in displacement_options else 0
            )

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
                'turning_radius': 5.2,
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
                'is_speed_alert': 'Yes',
                'ncap_rating': ncap_rating
            }
            
            input_df = pd.DataFrame([input_data])
            prob = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]
            
            st.markdown(f"### Claim Probability: **{prob:.2%}**")
            if pred == 1:
                st.error("High Risk – Likely to file a claim")
            else:
                st.success("Low Risk – Unlikely to file a claim")
            
            if prob > 0.15:
                st.warning("Suggest higher premium or further checks.")
            elif prob < 0.05:
                st.success("Great low-risk customer!")

# Sidebar footer
st.sidebar.markdown("### Car Insurance Analytics Dashboard")
st.sidebar.markdown("Built with Streamlit & Plotly | Model: LightGBM")