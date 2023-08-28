import streamlit as st
import pickle
import numpy as np
import pandas as pd

def show_details_page():
    st.write("#### Details")
    st.write("This is the head of the cleaned data frame. There are six features and one label. This is before encoding and scaling was performed. \n")
    

    st.image('cleanedData.png',caption = "Head of Cleaned Data Frame")
    st.write("- rob1_outcomeId: The label. 0 is for caught stealing and 1 is for a succesfully stolen base. ")  
    st.write("- sp: Runner stolen base percentage in 2016. This ended up being the most predictive feature, even with a good amount of Na's, and some players only having a couple attempts. Na's were filled with the median 'sp'.  ")    
    st.write("- csp: Percentage of allowed steals in 2016. Essentially how good the catcher was at throwing runners out. The second most predictive feature.") 
    st.write("- pitchType: Fastball, cutter, etc. ")
    st.write("- pitchSpeed: Speed of pitch at the plate. ")  
    st.write("- pitcZone: Location of pitch at the plate. Refer to 'Predict' for chart. ")  
    st.write("- outcomeDescription: Ball, Strike, Pitch Out, etc. ")  
    st.write(" ")
    st.write(" ")
 
    st.write(" The data is imbalanced, with more data points of runners stealing succesfully than being caught stealing."
              " This mirrors real life, with around 75% of runners succesfully stealing second on a given attempt in 2016. ")
    st.image('outcomeHistogram.png',caption = "Outcome Histogram")


    st.write("We can see from this pairplot that this outcome is hard to predict, but it is possible with these features.")
    st.write(" ")
    st.image('seabornPlot.png')
    st.write(" ")
    

    st.write("Logistic regression model with 'liblinear' solver gave me my best predictions. My goal was to maximize the f1 score.  ")
    
    st.write("Because the data is imbalanced, I had to compete against just picking 'success' everytime. I called this the 'baseline'.")
    st.write("To test my model I created 300 seperate train_test_splits. I recorded the f1 scores and accuracies for my model and for the baseline, and took the average. I believe this is called Monte-Carlo cross validation. I did this because I didn't have a huge dataset, and the metrics are largely affected by how the data is split into train/test. ")


    st.write("I wanted to maximize my f1 score; more specifically the difference between the baseline f1 and my models f1. ")
  

    st.write("My average f1 score was 0.865. If we were to just predict 'successfully stolen' everytime, we get an average f1 score of 0.845. The standard deviation for both is .011 for both groups. " )
    st.write("Not great, but it does beat the baseline f1 score.")

    st.write("My model had an average accuracy of 0.779 (guessing 77.9% of the test set right), while the baseline had an average of 0.732. The standard deviation for both is .017 for both groups. ")
    st.write("Not bad for such an difficult thing to predict!")
    st.write(" ")
    
    
    st.write("### Average F1 Score: 0.865")
    st.write("### Average Accuracy: 0.779")
    st.write("### Best F1 Score: 0.902")
    st.write("### Best Accuracy: 0.836")

    st.write(" ")
    st.write(" ")

    st.write("A link to the jupyter notebook is at https://github.com/SamHero16/CaughtOrNot/blob/main/CaughtOrNot.ipynb.")
    st.write(" ")
   
    
    
    
    st.write("Data aquired from: \n"
             " - https://www.kaggle.com/datasets/sportradar/baseball \n "
            " - https://www.fangraphs.com \n"
            " - https://baseballsavant.mlb.com/sprint_speed_leaderboard")
