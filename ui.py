#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:43:05 2020

@author: jordy
"""

import streamlit as st
import io
from model import Model
from numpy import uint8, fromstring, newaxis
from tensorflow.keras.applications.inception_v3 import preprocess_input
from cv2 import imdecode, resize

#The model which predicts ingredients on pizzas is loaded
pizza_detector = Model(weights_path = './weights') 

#Brief introduction
st.write("""

# Pizza Detector
Created by Jordy Dal Corso

---

Instructions to use this app:
    
* Upload the image of a pizza (.jpg file) in the box below.
* Wait for the app to detect its ingredients!
* Ingredients allowed are Pepperoni, Mushrooms, Onions, Peppers, Black Olives, Tomatoes and Basil

For more details about this project please refer to https://github.com/jdalcorso/pizza_detector
""")


#Image uploader Stramlit object
uploaded_file = st.file_uploader("Upload your pizza here:", type=['jpg'])

#Predictions are processed only if an image is loaded
if uploaded_file is not None:
    
    
    #In this first part of the script:
    #- the loaded image is processed
    #- the model is applied over the processed image 
    #- the ingredients over the pizza are predicted and saved
    
    
    #Converting the image from bytes to array
    img_stream = io.BytesIO(uploaded_file.read())
    img = imdecode(fromstring(img_stream.read(), uint8), 1)
    
    #BGR to RGB and resizing to get standard images as input for the model
    img_to_predict = resize(img, (299,299))
    
    #The line above because prediction needs 4dim array
    img_to_predict = img_to_predict[newaxis, ...]
    
    #The line below to convert into InceptionV3 standard input
    img_to_predict = preprocess_input(img_to_predict)
    
    #Applying the model to predict the toppings over the pizza provided
    predictions = pizza_detector.predict_this_image(img_to_predict)
    
    
    
    #In this second part of the script, predictions are visualized.
    #In particular the loaded image is plotted and the predictions are
    #written near it.
    #There is also a piace of information about how to read the predictions.
    
    
    #Dividing the streamlit view in 2 columns for a better visualization
    col1, col2 = st.beta_columns([0.7,1])
    
    with col1:
        img = img[...,::-1].copy()
        img = resize(img, (265,265))
        st.image(img)
        
    with col2:
        #do stuff, better add some visualz
        st.write('Pepperoni:   ',"{:.3f}".format(predictions[0,0]))
        st.write('Mushrooms: ',"{:.3f}".format(predictions[0,1]))
        st.write('Onions:      ',"{:.3f}".format(predictions[0,2]))
        st.write('Peppers:    ',"{:.3f}".format(predictions[0,3]))
        st.write('Olives:      ',"{:.3f}".format(predictions[0,4]))
        st.write('Tomatoes: ',"{:.3f}".format(predictions[0,5]))
        st.write('Basil:      ',"{:.3f}".format(predictions[0,6]))
    
    st.write('''
             How to read the results? 
             
             If the value of an ingredient is near 1, 
             the ingredient is probably on the pizza ''')
             
             

        
        
    
    
  
        
