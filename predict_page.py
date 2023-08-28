import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def load_model():
    with open('savedsteps.pkl','rb') as file:
        data  = pickle.load(file)

    return data

pkData = load_model()

classifier = pkData['model']
le_pitchType = pkData['le_pitchType']
le_OutcomeDescription = pkData['le_OutcomeDescription']
scaler = pkData['MinMaxScaler']


def show_predict_page():
    st.write("## Caught or Not?")
    st.write("""###### Predict if a runner would be safe or out if they attempted to steal second in a given scenario.""")
   
    


    catchersApp = [['A.J. Pierzynski', ': ',0.25], ["Custom Catcher" ': ', '---'],
    ['Wilson Ramos',': ', 0.37],
    ['David Ross',': ', 0.27],
    ['Hank Conger', ': ',0.19],
    ['Carlos Ruiz', ': ',0.42],
    ['Manny Piña', ': ',0.28],
    ['Robinson Chirinos', ': ',0.27],
    ['Dioner Navarro',': ', 0.23],
    ['Welington Castillo',': ', 0.38],
    ['Miguel Montero', ': ',0.11],
    ['Nick Hundley', ': ',0.14],
    ['Drew Butera',': ', 0.25],
    ['Jeff Mathis', ': ',0.3],
    ['René Rivera', ': ',0.3],
    ['Geovany Soto', ': ',0.19],
    ['Chris Stewart',': ', 0.27],
    ['Anthony Recker', ': ',0.22],
     ['Jose Lobaton',': ', 0.28],
    ['Matt Wieters',': ',0.35],
    ['Erik Kratz',': ', 0.47],
    ['Russell Martin', ': ',0.15],
    ['Brian McCann', ': ',0.23],
    ['Ryan Hanigan',': ', 0.28],
    ['Stephen Vogt',': ', 0.28],
    ['Sandy León', ': ',0.41],
    ['Francisco Cervelli',': ', 0.19],
    ['Tuffy Gosewisch', ': ',0.25],
    ['Austin Romine',': ', 0.17],
    ['Jarrod Saltalamacchia', ': ',0.24],
    ['Devin Mesoraco', ': ',0.27],
    ['A.J. Ellis',': ', 0.27],
    ['Bobby Wilson',': ', 0.17],
    ['Derek Norris', ': ',0.21],
    ['Martín Maldonado', ': ',0.4],
    ['Yadier Molina', ': ',0.21],
    ['Caleb Joseph',': ', 0.31],
    ['Juan Centeno',': ', 0.14],
    ['Salvador Perez',': ', 0.48],
    ['Chris Gimenez',': ', 0.26],
    ['Alex Avila',': ', 0.22],
    ["Travis d'Arnaud", ': ',0.22],
    ['Jonathan Lucroy', ': ',0.39],
    ['Eric Fryer', ': ',0.25],
    ['Kurt Suzuki', ': ',0.19],
    ['Chris Iannetta', ': ',0.31],
    ['Jason Castro', ': ',0.24],
    ['Tyler Flowers', ': ',0.05],
    ['Buster Posey', ': ',0.37],
    ['Chris Herrmann',': ',0.39],
    ['Josh Phegley', ': ',0.11],
    ['Dustin Garneau', ': ',0.38],
    ['Steve Clevenger', ': ',0.31],
    ['Yan Gomes', ': ',0.37],
    ['Josh Thole',': ',0.25],
    ['Christian Vázquez', ': ',0.35],
    ['Christian Bethancourt',': ', 0.27],
    ['Tucker Barnhart', ': ',0.33],
    ['Ramón Cabrera',': ', 0.21],
    ['John Ryan Murphy',': ', 0.38],
    ['Carlos Pérez', ': ',0.38],
    ['Evan Gattis',': ', 0.46],
    ['Cameron Rupp', ': ',0.27],
    ['Brett Nicholas',': ', 0.4],
    ['Bryan Holaday', ': ',0.38],
    ['Yasmani Grandal', ': ',0.29],
    ['Gary Sánchez',': ', 0.42],
    ['Tony Wolters',': ', 0.31],
    ['Willson Contreras',': ', 0.37],
    ['J.T. Realmuto', ': ',0.35],
    ['Jett Bandy',': ', 0.4],
    ['Curt Casali', ': ',0.36],
    ['James McCann',': ', 0.45],
    ['Mike Zunino', ': ',0.27],
    ['Omar Narváez', ': ',0.08],
    ['Luke Maile', ': ',0.28],
    ['Kevin Plawecki', ': ',0.24],
    ['Trevor Brown',': ', 0.23]]


    catchersApp2 = []
    for i in catchersApp:
        string=' '.join([str(item) for item in i]) + '%'
        catchersApp2.append(string)

    x = st.selectbox("Catcher (with percentage of runners succesfully thrown out at second in 2016)", catchersApp2)[-4:-1]

    if  x != '---':
        cspApp = 1 - float(x)
    elif x == '---':
        cspApp = 1 - st.slider("% of runners thrown out", 0.0,1.0, .5 )

    


    runnersApp = [['DJ LeMahieu',': ', 0.61],["Custom Runner" ': ', '---'],
    ['Daniel Murphy',': ', 0.62],
    ['Trea Turner',': ', 0.85],
    ['Jose Altuve',': ', 0.75],
    ['Joey Votto', ': ',0.89],
    ['José Peraza',': ', 0.68],
    ['Charlie Blackmon',': ', 0.65],
    ['Jean Segura',': ', 0.77],
    ['Mookie Betts', ': ',0.87],
    ['Dustin Pedroia',': ', 0.64],
    ['Cameron Maybin', ': ',0.71],
    ['Mike Trout',': ', 0.81],
    ['José Ramírez', ': ',0.76],
    ['Starling Marte', ': ',0.8],
    ['Yadier Molina', ': ',0.6],
    ['J.D. Martinez',': ', 0.33],
    ['Ryan Braun',': ', 0.76],
    ['J.T. Realmuto',': ', 0.75],
    ['Elvis Andrus',': ', 0.75],
    ['Freddie Freeman', ': ',0.86],
    ['Paulo Orlando',': ', 0.82],
    ['Hyun Soo Kim', ': ',0.25],
    ['Adam Frazier',': ', 0.8],
    ['Francisco Lindor', ': ',0.79],
    ['Devon Travis', ': ',0.8],
    ['Christian Yelich',': ', 0.69],
    ['Paul Goldschmidt', ': ',0.86],
    ['Tyler Naquin',': ', 0.67],
    ['Nolan Arenado',': ', 0.4],
    ['Xander Bogaerts',': ', 0.76],
    ['César Hernández', ': ',0.57],
    ['Kelby Tomlinson',': ', 0.83],
    ['Kris Bryant',': ', 0.62],
    ['Anthony Rizzo',': ', 0.38],
    ['Ender Inciarte',': ', 0.7],
    ['Brandon Phillips', ': ',0.64],
    ['Ichiro Suzuki',': ', 0.83],
    ['Ian Kinsler',': ', 0.7],
    ['Buster Posey',': ', 0.86],
    ['Eduardo Núñez', ': ',0.8],
    ['Lorenzo Cain',': ', 0.74],
    ['Odúbel Herrera', ': ',0.78],
    ['Hanley Ramirez', ': ',0.75],
    ['Jonathan Villar',': ', 0.78],
    ['Ian Desmond',': ', 0.78],
    ['Adam Eaton', ': ',0.74],
    ['Josh Donaldson', ': ',0.88],
    ['Josh Harrison',': ', 0.83],
    ['Norichika Aoki',': ', 0.44],
    ['Whit Merrifield',': ', 0.73],
    ['Tim Anderson',': ', 0.83],
    ['Mark Reynolds',': ', 0.33],
    ['Tyler Saladino',': ', 0.69],
    ['Jorge Polanco', ': ',0.57],
    ['Neil Walker',': ', 0.75],
    ['Josh Reddick',': ', 0.73],
    ['Andrelton Simmons', ': ',0.91],
    ['Asdrúbal Cabrera',': ', 0.83],
    ['Robbie Grossman',': ', 0.4],
    ['Yoenis Cespedes',': ', 0.75],
    ['Kyle Seager',': ', 0.75],
    ['C.J. Cron',': ', 0.4],
    ['Jarrod Dyson',': ', 0.81],
    ['Chris Owings',': ', 0.91],
    ['Ángel Pagán', ': ',0.79],
    ['Dexter Fowler',': ', 0.76],
    ['Chris Young',': ', 0.67],
    ['Didi Gregorius',': ', 0.88],
    ['Jason Kipnis',': ', 0.83],
    ['Justin Turner', ': ',0.8],
    ['Carlos Correa', ': ',0.81],
    ['Stephen Piscotty',': ', 0.58],
    ['Javier Báez', ': ',0.8],
    ['Adonis García',': ', 0.6],
    ['Hernán Pérez',': ', 0.83],
    ['Yasmany Tomás',': ',0.33],
    ['Ben Zobrist', ': ',0.6],
    ['Trevor Story',': ', 0.62],
    ['Rougned Odor',': ', 0.67],
    ['Kole Calhoun',': ', 0.4],
    ['Nick Franklin',': ', 0.86],
    ['Sean Rodríguez', ': ',0.67],
    ['Anthony Rendon',': ', 0.67],
    ['Eddie Rosario', ': ',0.71],
    ['Joey Rickard',': ', 0.8],
    ['Brian Dozier',': ', 0.9],
    ['Dee Strange-Gordon',': ', 0.81],
    ['Jackie Bradley Jr.',': ', 0.82],
    ['José Reyes', ': ',0.82],
    ['Jonathan Schoop',': ', 0.33],
    ['Kevin Pillar', ': ',0.7],
    ['Brandon Guyer',': ', 0.6],
    ['Eric Hosmer',': ', 0.62],
    ['Denard Span', ': ',0.63],
    ['Scott Schebler', ': ',0.33],
    ['Michael Bourn',': ', 0.75],
    ['Francisco Cervelli',': ', 0.75],
    ['Carlos Ruiz',': ', 0.75],
    ['Yasiel Puig',': ', 0.71],
    ['Jacoby Ellsbury', ': ',0.71],
    ['Scooter Gennett', ': ',0.89],
    ['Aaron Hill',': ', 0.67],
    ['Conor Gillaspie',': ', 0.33],
    ['Brett Gardner', ': ',0.8],
    ['George Springer',': ',0.47],
    ['Alcides Escobar',': ', 0.81],
    ['Billy Hamilton', ': ',0.88],
    ['Johnny Giavotella',': ', 0.57],
    ['Matt Szczur',': ', 0.33],
    ['Carlos Santana', ': ',0.71],
    ['Wil Myers', ': ',0.82],
    ['Ketel Marte',': ', 0.69],
    ['Tony Wolters', ': ',0.8],
    ['Matt Duffy',': ', 0.62],
    ['Ronald Torreyes',': ', 0.67],
    ['Gregory Polanco', ': ',0.74],
    ['Alexi Amarista',': ', 0.82],
    ['Alex Dickerson', ': ',0.83],
    ['Domingo Santana',': ', 0.4],
    ['Andrew McCutchen',': ', 0.46],
    ['Brock Holt',': ',0.57],
    ['José Iglesias', ': ',0.64],
    ['Jung Ho Kang', ': ',0.75],
    ['Howie Kendrick',': ', 0.83],
    ['Austin Jackson', ': ',0.67],
    ['Marwin Gonzalez',': ', 0.67],
    ['Iván De Jesús Jr.',': ', 0.75],
    ['Yonder Alonso', ': ',0.75],
    ['Michael Saunders',': ', 0.33],
    ['Gerardo Parra', ': ',0.6],
    ['Andrés Blanco', ': ',0.4],
    ['Zack Cozart',': ', 0.8],
    ['Peter Bourjos',': ', 0.6],
    ['Evan Gattis', ': ',0.67],
    ['Chase Headley', ': ',0.8],
    ['Jay Bruce',': ', 0.67],
    ['Rob Refsnyder',': ', 0.67],
    ['Rajai Davis',': ', 0.88],
    ['Jake Lamb',': ', 0.86],
    ['Ezequiel Carrera',': ', 0.64],
    ['Brett Lawrie',': ', 0.7],
    ['Eugenio Suárez', ': ',0.69],
    ['Tim Beckham',': ', 0.67],
    ['Miguel Rojas',': ', 0.67],
    ['Leonys Martin', ': ',0.8],
    ['Khris Davis',': ', 0.33],
    ['Steven Souza Jr.',': ', 0.54],
    ['Joc Pederson',': ', 0.75],
    ['Kevin Kiermaier', ': ',0.88],
    ['Justin Upton', ': ',0.69],
    ['Travis Jankowski', ': ',0.71],
    ["Chase d'Arnaud",': ', 0.75],
    ['Luis Sardinas',': ', 0.67],
    ['Jayson Werth', ': ',0.83],
    ['Erick Aybar', ': ',0.38],
    ['Bryce Harper', ': ',0.68],
    ['Brad Miller',': ', 0.6],
    ['Travis Shaw',': ', 0.83],
    ['Shin-Soo Choo', ': ',0.67],
    ['Keon Broxton', ': ',0.85],
    ['Freddy Galvis',': ', 0.74],
    ['Adam Duvall',': ', 0.55],
    ['Alexei Ramirez', ': ',0.47],
    ['Danny Santana', ': ',0.57],
    ['Randal Grichuk',': ', 0.56],
    ['Juan Lagares',': ', 0.67],
    ['Jurickson Profar',': ',0.67],
    ['Mike Napoli', ': ',0.83],
    ['Mallex Smith', ': ',0.67],
    ['Addison Russell', ': ',0.83],
    ['Logan Morrison', ': ',0.67],
    ['Jake Smolinski',': ', 0.33],
    ['B.J. Upton', ': ',0.77],
    ['Marcus Semien', ': ',0.83],
    ['Curtis Granderson',': ', 0.67],
    ['Eduardo Escobar',': ', 0.25],
    ['Jeremy Hazelbaker',': ', 0.71],
    ['Max Kepler',': ', 0.75],
    ['Billy Burns',': ', 0.77],
    ['Tyler Holt', ': ',0.57],
    ['Rafael Ortega', ': ',0.73],
    ['Carlos Gómez',': ', 0.78],
    ['Coco Crisp', ': ',0.67],
    ['Russell Martin', ': ',0.67],
    ['Michael A. Taylor',': ', 0.82],
    ['Jason Heyward', ': ',0.73],
    ["Shawn O'Malley", ': ',0.75],
    ['Yasmani Grandal', ': ',0.25],
    ['Christian Bethancourt',': ', 0.33],
    ['David Wright', ': ',0.6],
    ['Scott Van Slyke', ': ',0.33],
    ['Todd Frazier', ': ',0.75],
    ['Byron Buxton', ': ',0.83],
    ['Trayce Thompson', ': ',0.83],
    ['Grégor Blanco', ': ',0.67],
    ['Chris Carter', ': ',0.75],
    ['Nolan Reimold',': ', 0.33],
    ['Jimmy Rollins', ': ',0.71],
    ['Michael Conforto',': ', 0.67],
    ['Alex Gordon',': ', 0.89],
    ['Brandon Barnes',': ', 0.33],
    ['Nick Ahmed',': ', 0.71],
    ['Ryan Zimmerman',': ', 0.8],
    ['Cristhian Adames', ': ',0.4],
    ['Ben Revere', ': ',0.74],
    ['Aaron Hicks',': ', 0.43],
    ['Tony Kemp', ': ',0.67],
    ['A.J. Ellis',': ', 0.67],
    ['Cody Asche',': ', 0.75],
    ['Jason Castro',': ', 0.67],
    ['Danny Espinosa',': ', 0.82],
    ['Jake Marisnick', ': ',0.67],
    ['Kirk Nieuwenhuis', ': ',0.47],
    ['Delino DeShields',': ', 0.73],
    ['Dioner Navarro', ': ',0.33],
    ['Colby Rasmus',': ', 0.8],
    ['Alejandro De Aza', ': ',0.57],
    ['Aaron Altherr', ': ',0.78],
    ['Chris Coghlan',': ', 0.67],
    ['Derek Norris',': ', 0.82],
    ['Adalberto Mondesi',': ', 0.9],
    ['Shane Robinson',': ', 0.6],
    ['Ji Man Choi',': ', 0.33]]

    runnersApp2 = []
    for i in runnersApp:
        string=' '.join([str(item) for item in i]) + '%'
        runnersApp2.append(string)

    y = st.selectbox("Runner (with percentage of succesfully stolen bases in 2016)", runnersApp2)[-4:-1]

 

    if  y != '---':
        spApp = float(y)
    elif y == '---':
        spApp = st.slider("% of bases succesfully stolen", 0.0,1.0, .5 )


   

    
    ptApp = ['Cutter (CU)', 'Fastball (FA)', 'Sinker (SI)', 'Changedup (CH)', 'Slider (SL)', 'Splitter (SP)', 'Cutter (CT)', 'Pitch Out (PI)', 'Knuckleball (KN)']

    pitchTypeApp = st.selectbox("Pitch Type", ptApp)[-3:-1]




    odAppNP = [ 'Strike Swinging', 'Dirt Ball', 'Strike Looking', 'Ball', 'Foul Tip'] 
             
    
    
    if pitchTypeApp != 'PI':
        outcomeDescriptionApp = st.selectbox("Pitch Resolution", odAppNP)
    else:
        outcomeDescriptionApp = st.selectbox("Pitch Resolution", ['Pitch Out'])

    





    
    pitchSpeedAPP = st.slider("Pitch Speed (MPH)" , 70,100, 85 )




    st.image('PitchZonesIMG.png',caption = "Pitch Zones Diagram")

    pzApp = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    pzApp2 = [11,13]
    pzApp3 = [10,11,12,13]

    
    if pitchTypeApp != 'PI':
        if outcomeDescriptionApp == 'Dirt Ball':
            pitchZoneApp = st.selectbox("Pitch Zone", pzApp3)
        else:
            pitchZoneApp = st.selectbox("Pitch Zone", pzApp)
    
    else:
        pitchZoneApp = st.selectbox("Pitch Zone", pzApp2)
        
    


   
    

#sp	csp	pitchType	pitchSpeed	pitchZone	outcomeDescription


    ok = st.button("Steal!")
    if ok:
        
        X =np.array([[spApp,cspApp,pitchTypeApp,pitchSpeedAPP,pitchZoneApp,outcomeDescriptionApp]])

        X[:,5] = le_OutcomeDescription.transform(X[:,5])
        X[:,2] = le_pitchType.transform(X[:,2])
        X[:,3] = scaler.transform(X[:,3].reshape(1, -1))
        X = X.astype(float)

        prediction = classifier.predict(X)
        if prediction == 1:
            st.subheader("Safe! Keep on stealing...")
        if prediction == 0:
            st.subheader("Out! Too slow...")

        pred = classifier.predict_proba(X)[0][prediction]

        st.subheader("Probability: " + str(pred[0].round(2)))


        
        
        

        

