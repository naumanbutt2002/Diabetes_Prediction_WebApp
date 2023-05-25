# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:52:17 2023

@author: nauma
"""


import numpy as np
import pickle as pk
import streamlit as st

loaded_model=pk.load(open('C:/Users/nauma/Desktop/ai_project_final/trained_diabetes_model.sav','rb'))#read binary


def Prediction_Function(input_data):
    #Making a Predictive System
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance (row)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
def main():
    st.title('Diabetes Prediction System')
    children=st.number_input('Number of Childrens')
    Glucose=st.number_input('Glucose Level')
    Blood_Pressure=st.number_input('Blood Pressure Value')
    Skin_Thickness=st.number_input('Skin Thickness Value')
    Insulin=st.number_input('Insulin Value')
    BMI=st.number_input('BMI Value')
    Pedigree_Func=st.number_input('Diabetes Pedigree Function')
    Age=st.number_input('Enter Your Age')
    
    prediction=''
    
    if st.button('Diabetes Test Result'):
        prediction=Prediction_Function([children,Glucose,Blood_Pressure,Skin_Thickness,Insulin,BMI,Pedigree_Func,Age])
    
    st.success(prediction)
    
if __name__=='__main__':
    main()






















        