import streamlit as st
import pickle 
import pandas as pd

#reading the encoder, model and scaler object files
encoder = pickle.load(open("encoder.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

#setting the title and text
st.title("🌼Iris Flower Classification")
st.write("*Developed for 🌎 with ❤️‍🔥 by Sohaib👨🏻‍💻🇵🇰*")

#taking the input from user
new_SL = st.number_input("Enter sepalLength (cm):", min_value=0.0, max_value=10.0, step=0.1)
new_SW = st.number_input("Enter sepalWidth (cm):", min_value=0.0, max_value=10.0, step=0.1)
new_PL = st.number_input("Enter petalLength (cm):", min_value=0.0, max_value=10.0, step=0.1)
new_PW = st.number_input("Enter petalWidth (cm):", min_value=0.0, max_value=10.0, step=0.1)

#button to trigger the classification
if st.button("Classify"):
    new_value = pd.DataFrame([[new_SL, new_SW, new_PL, new_PW]])
    new_value_scaled = scaler.transform(new_value)
    prediction = model.predict(new_value_scaled)
    finalans = encoder.inverse_transform(prediction)
    st.markdown(f"Prediction result: **{finalans[0]}**")    
