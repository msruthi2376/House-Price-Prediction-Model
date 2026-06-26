import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")

# 2. Data Loading & Model Training
@st.cache_resource
def get_model():
    # Ensure Housing.csv is in the same directory as this script
    df = pd.read_csv("Housing.csv")
    le_dict = {}
    # Encode categorical variables
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model, le_dict

model, le_dict = get_model()

# 3. Sidebar Input Controls
st.sidebar.header("🏠 Property Specifications")
with st.sidebar:
    area = st.number_input("Area (sq ft)", min_value=1000, value=5000)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    stories = st.slider("Stories", 1, 4, 2)
    parking = st.slider("Parking Spaces", 0, 3, 1)
    
    mainroad = st.selectbox("Mainroad Access", ["yes", "no"])
    guestroom = st.selectbox("Guestroom Available", ["yes", "no"])
    basement = st.selectbox("Basement Available", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# 4. Main UI Layout
st.title("🏡 Smart Housing Price Estimator")
st.markdown("### Predict the market value of your property in seconds.")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("How it works")
    st.write("This application uses a **Random Forest Regressor** to analyze historical housing data. "
             "The model evaluates features like square footage, location, and amenities to estimate a fair market price.")
    
    if st.button("🚀 Calculate Estimated Price", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'area': [area], 'bedrooms': [bedrooms], 'bathrooms': [bathrooms], 
            'stories': [stories], 'parking': [parking],
            'mainroad': [le_dict['mainroad'].transform([mainroad])[0]],
            'guestroom': [le_dict['guestroom'].transform([guestroom])[0]],
            'basement': [le_dict['basement'].transform([basement])[0]],
            'hotwaterheating': [le_dict['hotwaterheating'].transform([hotwaterheating])[0]],
            'airconditioning': [le_dict['airconditioning'].transform([airconditioning])[0]],
            'prefarea': [le_dict['prefarea'].transform([prefarea])[0]],
            'furnishingstatus': [le_dict['furnishingstatus'].transform([furnishingstatus])[0]]
        })
        
        # Enforce exact column order
        feature_order = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
                         'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
        input_data = input_data[feature_order]
        
        prediction = model.predict(input_data)
        
        # Display Results
        st.metric(label="Estimated Property Price", value=f"₹{round(prediction[0], 2):,}")
        st.balloons()

with col2:
    st.info("💡 **Pro Tip:** Houses with Air Conditioning and more parking spaces generally have a higher impact on the final valuation.")