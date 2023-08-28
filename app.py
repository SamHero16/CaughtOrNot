import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from details_page import show_details_page


q = st.sidebar.selectbox("Menu" , ("Predict","Background","Details"))

if q == "Predict":
    show_predict_page()


elif q == "Background":
    show_explore_page()

elif q == "Details":
    show_details_page()
   
