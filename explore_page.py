import streamlit as st
import pickle
import numpy as np
import pandas as pd

def show_explore_page():
    st.write("#### Background")
    st.write(" This project combines pitch data as well as "
                "2016 player stats to predict if a runner would be safe or thrown out when attempting to steal second (in 2016)."
                " I wanted to create a predictor based on pitch data (ie. pitch speed, pitch type, pitch location, etc.), as well as catcher and runner data: all things that could be easily gathered after a pitch is thrown."
                " I was not able to find any data on runner lead distance, which I would have like to.")
    
    
    st.write("  Finding the data I needed was not easy, but a 2016 SportsRadar dataset ended up having most of what I needed."
                " Each row was a pitch event with pitch data where a runner had attempted to steal. I narrowed it down to runners who had tried to steal second."
                " I then found stats on the specific runners and catchers seen in each event of that dataset.")

    st.write("  Most of my challenges came from gathering and combining the multiple data sets."
             " An interesting problem I encountered was that Sports Radar does not include the names of their players, but just "
              "a 'SportsRadarID'. This 'SportsRadarID' exists no where on the internet, except the costly Sports Radar API.  So to get the names of the players, I had to get trial-keys for the Sports Radar API,  "
              "which only allows a certain number of requests.")
    
    st.write("  Another issue that arose is that alot of the features that I had believed would be predictive were not. "
             "At one point I was including runner speed and catcher rSB in my model, and come to find out they are "
             "not helpful at all. I also felt setback when my initial tests of my model did worse than just guessing the most popular outcome ('safe'). "
             "But after a lot of tweaking, adjustment and testing, my model can do better than that baseline.")
    
    st.write(f' Stealing second base is a very hard thing to predict, and I realized the reality of that more and more as the project went along'
             '. This project was very teaching: \n - Gathering the right data is all important \n - Trial and error is crucial '
             " \n - Sometimes what you think will work, won't ")
    
    