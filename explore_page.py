import streamlit as st
import pickle
import numpy as np
import pandas as pd

def show_explore_page():
    st.write("#### Background")
    st.write(" This is a personal passion project of mine created in Summer 2023. I wanted to practice my skills and get experience, while diving deep into something I love: baseball.")
    st.write("The goal of this project is to "
                "predict if a runner would be safe or thrown out when attempting to steal second. Stealing is extremely nuanced and notoriously hard to quantify and predict, so this is no easy task."
                " I wanted to create a predictor based on pitch data (ie. pitch speed, pitch type, pitch location, etc.), as well as catcher and runner data: all things that could be easily gathered after a pitch is thrown."
                " The perspective of the project is someone who has seen the players on the field, seen the pitch, and then asked to predict if the runner was thrown out or safe at second."
                " "
                
                )
    
    
    st.write("  Finding the data I needed was not easy, but a 2016 SportsRadar dataset ended up having most of what I needed."
                " Each row was a pitch event with pitch data where a runner had attempted to steal. I narrowed it down to runners who had tried to steal second."
                " I then found stats on the specific runners and catchers seen in each event of that dataset."
                "I was not able to find any data on runner lead distance, which I would have like to.")

    st.write("  A lot of challenges came from gathering and combining the multiple data sets."
             " An interesting problem I encountered was that Sports Radar does not include the names of their players, but just "
              "a 'SportsRadarID'. This 'SportsRadarID' exists no where on the internet, except the costly Sports Radar API.  So to get the names of the players, I had to get trial-keys for the Sports Radar API,  "
              "which only allows a certain number of requests.")
    
    st.write("  Another issue that arose is that alot of the features that I had believed would be predictive were not. "
             "At one point I was including runner speed, catcher 'rSB', batter and catcher handedness data, and more in my model; come to find out they are "
             "not helpful at all. I also felt setback when my initial tests of my model did worse than just guessing the most popular outcome ('safe'). "
             "But after a lot of tweaking, adjustment and testing, my model can do better than that baseline.")
    
    st.write(f' Stealing second base is a very hard thing to predict, and I realized the reality of that more and more as the project went along')
             
    st.write(" I will also say that the Jupyter Notebook does not tell the full story of the trial and error. I cleaned it up so it was readable, but there were hours and hours of tweaking and testing different models, features, strategies to fill NA's, dealing with outliers, etc.")
    
    