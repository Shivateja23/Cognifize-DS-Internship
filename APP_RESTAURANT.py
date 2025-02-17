import streamlit as st
import pandas as pd
import pickle

# Set the page layout to wide (optional)
st.set_page_config(layout="wide")

# Load your trained model (ensure that it was trained on the 12 features below)
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Restaurant Rating Predictor")

# Display an image in the main area using the new parameter
st.image("restaurent2 img.jpg", use_container_width=True)

# Sidebar inputs (must exactly match the feature names and order used during training)
st.sidebar.header("Enter Restaurant Details:")

restaurant_id = st.sidebar.number_input("Restaurant ID", value=6317637)
country_code = st.sidebar.selectbox("Country Code", [162, 30, 216, 14, 37, 184, 214, 1, 94, 148, 215, 166, 189, 191, 208])
average_cost_for_two = st.sidebar.number_input("Average Cost for Two", value=1100)
has_table_booking = st.sidebar.selectbox("Has Table booking?", [0, 1], index=1)
has_online_delivery = st.sidebar.selectbox("Has Online delivery?", [0, 1], index=1)
is_delivering_now = st.sidebar.selectbox("Is delivering now?", [0, 1], index=0)
switch_to_order_menu = st.sidebar.selectbox("Switch to order menu?", [0, 1], index=0)
price_range = st.sidebar.number_input("Price range", value=3)
votes = st.sidebar.number_input("Votes", value=314)
cuisine_count = st.sidebar.number_input("Cuisine Count", value=3)
multiple_cuisines = st.sidebar.selectbox("Multiple Cuisines?", [0, 1], index=1)
has_table_booking_and_online_delivery = st.sidebar.selectbox("Has_Table_Booking_and_Online_Delivery?", [0, 1], index=0)

# When the user clicks "Predict Rating", build the input DataFrame and make a prediction
if st.sidebar.button("Predict Rating"):
    input_data = pd.DataFrame({
        "Restaurant ID": [restaurant_id],
        "Country Code": [country_code],
        "Average Cost for two": [average_cost_for_two],
        "Has Table booking": [has_table_booking],
        "Has Online delivery": [has_online_delivery],
        "Is delivering now": [is_delivering_now],
        "Switch to order menu": [switch_to_order_menu],
        "Price range": [price_range],
        "Votes": [votes],
        "Cuisine Count": [cuisine_count],
        "Multiple Cuisines": [multiple_cuisines],
        "Has_Table_Booking_and_Online_Delivery": [has_table_booking_and_online_delivery]
    })

    # Make a prediction using the trained model
    prediction = model.predict(input_data)

    # Display the predicted rating in a large font using custom HTML.
    st.markdown(
        f"<h1 style='text-align: center; font-size:72px; color: #4CAF50;'>Predicted Rating: {prediction[0]}</h1>",
        unsafe_allow_html=True
    )
