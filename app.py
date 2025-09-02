import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import requests
import json
import base64

# --- Configuration & Styling ---
st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

# Custom CSS for a futuristic, dark theme
def set_styles():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0c0d12;
            color: #d1d1e0;
        }
        .stButton>button {
            background-color: #7b2cbf;
            color: white;
            border-radius: 20px;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            background-color: #9d4edd;
            transform: scale(1.05);
            transition: all 0.2s ease-in-out;
        }
        h1, h2, h3 {
            color: #d1d1e0;
            text-align: center;
        }
        h1 {
            border-bottom: 2px solid #5a189a;
            padding-bottom: 10px;
        }
        .stExpander {
            border-radius: 15px;
            border: 1px solid #2a1a3a;
            background-color: #1a1b24;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .stInfo {
            background-color: #2a2a40;
            color: #a0a0ff;
            border-radius: 10px;
            padding: 10px;
        }
        .stNumberInput, .stSelectbox {
            background-color: #1a1b24;
            border: 1px solid #3d3d5c;
            border-radius: 10px;
        }
        .stSlider .st-eg {
            background: linear-gradient(90deg, #8a2be2, #00bfff);
        }
        .stCheckbox span {
            color: #a0a0ff;
        }
        </style>
    """, unsafe_allow_html=True)
set_styles()

# --- Model & Image Loading ---
MODEL_PATH = 'Best_tuned_optimal_restaurant_rating_model_GradientBoosting.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file '{MODEL_PATH}' not found.")
        st.stop()
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

pipeline = load_model()

@st.cache_data(show_spinner="Generating a beautiful futuristic restaurant...")
def generate_restaurant_image():
    # Placeholder for a static image for simplicity. 
    # In a real-world scenario, you would replace this with your API call.
    img_path = 'futuristic_restaurant_header.png'
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

# --- Define Features and Lists ---
ALL_FEATURE_COLUMNS = [
    'Longitude', 'Latitude', 'Average Cost for two', 'Votes', 'Distance',
    'Price range', 'Restaurant Name Length', 'Address Length',
    'Cuisine Count', 'Multiple Cuisines', 'Has Table booking',
    'Has Online delivery', 'Is delivering now', 'Has_Table_Booking_and_Online_Delivery',
    'Country Code', 'City', 'Currency'
]

COUNTRY_CODES_DICT = {
    162: "New Zealand", 30: "Brazil", 216: "United States", 14: "Australia",
    37: "Canada", 184: "South Africa", 214: "United Arab Emirates", 1: "India",
    94: "Indonesia", 148: "Philippines", 215: "United Kingdom", 166: "Qatar",
    189: "Singapore", 191: "Srilanka", 208: "Turkey"
}

PRICE_RANGES_DICT = {
    1: "Low", 2: "Medium", 3: "High", 4: "Very High"
}

CITIES = ['Makati City', 'Mandaluyong City', 'Pasay City', 'Pasig City', 'Quezon City', 'San Juan City', 'Santa Rosa', 'Tagaytay City', 'Taguig City', 'Brasília', 'Rio de Janeiro', 'São Paulo', 'Albany', 'Armidale', 'Athens', 'Augusta', 'Balingup', 'Beechworth', 'Boise', 'Cedar Rapids/Iowa City', 'Chatham-Kent', 'Clatskanie', 'Cochrane', 'Columbus', 'Consort', 'Dalton', 'Davenport', 'Des Moines', 'Dicky Beach', 'Dubuque', 'East Ballina', 'Fernley', 'Flaxton', 'Forrest', 'Gainesville', 'Hepburn Springs', 'Huskisson', 'Inverloch', 'Lakes Entrance', 'Lakeview', 'Lincoln', 'Lorn', 'Macedon', 'Macon', 'Mayfield', 'Mc Millan', 'Middleton Beach', 'Miller', 'Monroe', 'Montville', 'Ojo Caliente', 'Orlando', 'Palm Cove', 'Paynesville', 'Penola', 'Pensacola', 'Phillip Island', 'Pocatello', 'Potrero', 'Princeton', 'Rest of Hawaii', 'Savannah', 'Singapore', 'Sioux City', 'Tampa Bay', 'Tanunda', 'Trentham East', 'Valdosta', 'Vernonia', 'Victor Harbor', 'Vineland Station', 'Waterloo', 'Weirton', 'Winchester Bay', 'Yorkton', 'Abu Dhabi', 'Dubai', 'Sharjah', 'Agra', 'Ahmedabad', 'Allahabad', 'Amritsar', 'Aurangabad', 'Bangalore', 'Bhopal', 'Bhubaneshwar', 'Chandigarh', 'Chennai', 'Coimbatore', 'Dehradun', 'Faridabad', 'Ghaziabad', 'Goa', 'Gurgaon', 'Guwahati', 'Hyderabad', 'Indore', 'Jaipur', 'Kanpur', 'Kochi', 'Kolkata', 'Lucknow', 'Ludhiana', 'Mangalore', 'Mohali', 'Mumbai', 'Mysore', 'Nagpur', 'Nashik', 'New Delhi', 'Noida', 'Panchkula', 'Patna', 'Puducherry', 'Pune', 'Ranchi', 'Secunderabad', 'Surat', 'Vadodara', 'Varanasi', 'Vizag', 'Bandung', 'Bogor', 'Jakarta', 'Tangerang', 'Auckland', 'Wellington City', 'Birmingham', 'Edinburgh', 'London', 'Manchester', 'Doha', 'Cape Town', 'Inner City', 'Johannesburg', 'Pretoria', 'Randburg', 'Sandton', 'Colombo', 'Ankara', 'Istanbul']
CURRENCIES = ['Botswana Pula(P)', 'Brazilian Real(R$)', 'Dollar($)', 'Emirati Diram(AED)', 'Indian Rupees(Rs.)', 'Indonesian Rupiah(IDR)', 'NewZealand($)', 'Pounds(£)', 'Qatari Rial(QR)', 'Rand(R)', 'Sri Lankan Rupee(LKR)', 'Turkish Lira(TL)']

# --- Streamlit App Interface ---
st.title("🍽️ Restaurant Rating Prediction")

# Display the futuristic image below the header
header_image_data = generate_restaurant_image()
if header_image_data:
    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{header_image_data}" style="width: 80%; border-radius: 15px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);"></div>',
        unsafe_allow_html=True
    )

st.markdown("Enter the attributes of a restaurant below to get a predicted `Aggregate rating`.")

st.markdown("---")

with st.expander("Enter Restaurant Details", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location & Basic Info")
        country_code_options = [f"{code} ({name})" for code, name in COUNTRY_CODES_DICT.items()]
        selected_country_option = st.selectbox("Country Code", options=country_code_options)
        country_code = int(selected_country_option.split(' ')[0])
        
        city = st.selectbox("City", options=CITIES)
        
        st.markdown("**Location Coordinates**")
        longitude = st.number_input("Longitude", value=77.0813, format="%.6f", help="Location of the restaurant")
        latitude = st.number_input("Latitude", value=28.4787, format="%.6f", help="Location of the restaurant")
        distance = st.number_input("Distance (from centroid)", value=50.0, format="%.2f", min_value=0.0)

    with col2:
        st.subheader("💰 Cost & Features")
        st.markdown("---")
        average_cost_for_two = st.number_input("Average Cost for two", value=500, min_value=0)
        
        st.markdown("---")
        currency = st.selectbox("Currency", options=CURRENCIES)
        
        st.markdown("---")
        st.markdown("**Price Range**")
        price_range = st.radio("Select a Price Range:", options=list(PRICE_RANGES_DICT.keys()), format_func=lambda x: PRICE_RANGES_DICT[x])
        
        st.markdown("---")
        st.markdown("**Votes**")
        votes = st.slider("Votes", min_value=0, max_value=2000, value=100)
    
    with col3:
        st.subheader("⚙️ Engineered Features & Services")
        restaurant_name_length = st.slider("Restaurant Name Length", min_value=1, max_value=50, value=15)
        address_length = st.slider("Address Length", min_value=1, max_value=100, value=50)
        cuisine_count = st.slider("Cuisine Count", min_value=1, max_value=10, value=1)
        multiple_cuisines = 1 if cuisine_count > 1 else 0
        st.info("The 'Multiple Cuisines' feature is automatically calculated.")
        
        st.markdown("---")
        st.markdown("**Service Availability**")
        col_service1, col_service2, col_service3 = st.columns(3)
        with col_service1:
            has_table_booking = st.checkbox("Has Table Booking", value=True)
        with col_service2:
            has_online_delivery = st.checkbox("Has Online Delivery", value=False)
        with col_service3:
            is_delivering_now = st.checkbox("Is Delivering Now", value=False)
        
        has_table_booking_and_online_delivery = int(has_table_booking and has_online_delivery)

# --- Prediction Button & Result ---
st.markdown("---")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("Predict Rating"):
    if pipeline is not None:
        input_data = {
            'Longitude': longitude, 'Latitude': latitude, 'Average Cost for two': average_cost_for_two,
            'Votes': votes, 'Distance': distance, 'Price range': price_range,
            'Restaurant Name Length': restaurant_name_length, 'Address Length': address_length,
            'Cuisine Count': cuisine_count, 'Multiple Cuisines': multiple_cuisines,
            'Has Table booking': int(has_table_booking), 'Has Online delivery': int(has_online_delivery),
            'Is delivering now': int(is_delivering_now),
            'Has_Table_Booking_and_Online_Delivery': has_table_booking_and_online_delivery,
            'Country Code': country_code, 'City': city, 'Currency': currency
        }
        input_df = pd.DataFrame([input_data])
        input_df = input_df[ALL_FEATURE_COLUMNS]

        try:
            prediction = pipeline.predict(input_df)[0]
            st.success(f"**Predicted Aggregate Rating:** {prediction:.2f} ⭐")
            st.balloons()
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #5a5a7a;'>Developed with ❤️ using Streamlit and scikit-learn.</div>", unsafe_allow_html=True)