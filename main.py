import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# Function to generate dummy variables for the selected month


def generate_dummy_for_region(user_input):
    regions = ['Region_ashanti',
       'Region_brong ahafo', 'Region_central', 'Region_eastern',
       'Region_greater accra', 'Region_northern', 'Region_upper east',
       'Region_upper west', 'Region_volta', 'Region_western']
    # Check if user input is a valid month
    if user_input in regions:
        # Create a dictionary with all months as 0 except the selected month as 1
        dummy_dict = {region: 0 for region in regions}
        dummy_dict[user_input] = 1

        # Convert the dictionary to a DataFrame
        dummy_df = pd.DataFrame(dummy_dict, index=[0])
        return dummy_df

def generate_dummy_for_commodity(user_input):
    commoditys = ['Commodity_cocoyam',
       'Commodity_maize', 'Commodity_millet', 'Commodity_rice',
       'Commodity_sorghum', 'Commodity_yam']
    # Check if user input is a valid month
    if user_input in commoditys:
        # Create a dictionary with all months as 0 except the selected month as 1
        dummy_dict = {commodity: 0 for commodity in commoditys}
        dummy_dict[user_input] = 1

        # Convert the dictionary to a DataFrame
        dummy_df = pd.DataFrame(dummy_dict, index=[0])
        return dummy_df

def preprocess_inputs(year, area, rainfall, commodity, region):
    regions = generate_dummy_for_region(region)
    commoditys = generate_dummy_for_commodity(commodity)
    preprocessed_inputs = np.concatenate([year, [area, rainfall], regions.values, commoditys.values], axis=None)
    return preprocessed_inputs


# Define the user interface
st.title('Crop Yield Predictor')

year = st.slider('Year', min_value=2008, max_value=2040, step=1)  
area = st.slider('Area (HA)', min_value=0.0, max_value=10000.0, step=1.0)
rainfall = st.slider('Rainfall per district (mm/HA)', min_value=0.0, max_value=10.0, step=0.01)
commodity = st.selectbox('Commodity', ['Commodity_cocoyam',
       'Commodity_maize', 'Commodity_millet', 'Commodity_rice',
       'Commodity_sorghum', 'Commodity_yam'])  
region = st.selectbox('Region', ['Region_ashanti',
       'Region_brong ahafo', 'Region_central', 'Region_eastern',
       'Region_greater accra', 'Region_northern', 'Region_upper east',
       'Region_upper west', 'Region_volta', 'Region_western'])

if st.button('Predict'):
    model = joblib.load("model.joblib")
    # Preprocess inputs
    preprocessed_inputs = preprocess_inputs(year,area,rainfall,commodity,region)
    # Make prediction
    prediction = model.predict(preprocessed_inputs.reshape(1, -1))
    st.write(f"Predicted Yield: {prediction[0]:2f} MT/HA")




