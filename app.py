# Imports
import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import joblib
from scipy.stats import boxcox

# function to transform user input to suitable format expected by SVM model
def transform_input(df_encoded, lambdas):
    # Adding a small constant to 'oldpeak' to make all values positive
    df_encoded['oldpeak'] = df_encoded['oldpeak'] + 0.001

    # Apply Box-Cox Transformation
    for column, lambda_val in lambdas.items():
        # Ensure the column exists in the DataFrame
        if column in df_encoded.columns:
            # Apply the saved Box-Cox lambda to the input data
            df_encoded[column] = boxcox(df_encoded[column], lmbda=lambdas[column]) 
            # df_encoded[column] = (df_encoded[column] ** lambda_val - 1) / lambda_val
            print(f'{column} transformed using saved lambda')
        else:
            print(f'{column} not found in DataFrame')

    return df_encoded   # return transformed dataframe


# set page configuration
st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤",
        layout="wide"
    )

# declare containers
header = st.container()
content = st.container()

# blank line
st.write("")
 
# header container
with header:   
    # title and description
    st.title('Predicting Heart Disease')
    st.markdown('##### Model to determine if a patient has heart disease')

    # Header
    st.header("Heart Disease", divider="violet")

# content container
with content:
    main_col1, main_col2 = st.columns([7, 5])
    # main column 1
    with main_col1:
        # form to get user input for patient data
        with st.form("Preidct"):
            col1, col2, col3 = st.columns(3)    # define 3 columns
            
            # column 1
            with col1:
        
                age = st.number_input('Age in Years (age)', 0, 100, 0, 1)
                
                trestbps = st.number_input('Resting Blood Pressure (trestbps)', 80, 250, 80, 1)
                
                chol = st.number_input('Serum Cholestoral in mg/dl (chol)', 100, 600, 100, 1)
                
                thalach = st.number_input('Max Heart Rate Achieved (thalach)', 50, 250, 50, 1)
            
            # column 2
            with col2:
                
                oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest (oldpeak)', 0.0, 8.0, 0.0, 0.1)
        
                sex = st.selectbox("Sex (sex)", options=["Female", "Male"], index=0)
                
                cp = st.selectbox("Select Chest Pain Type (cp)", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], index=0)
                
                fbs = st.selectbox("Fasting Blood Sugar (fbs)", options=["Less Than 120 mg/dl", "Greater Than 120 mg/dl"], index=0)
                
                restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=["Normal", "Abnormal", "Ventricular Hypertrophy"], index=0)
                
            # column 3
            with col3:
                
                exang = st.selectbox("Exercise Induced Angina (exang)", options=["Yes", "No"], index=0)
                
                slope = st.selectbox("Select Slope of Peak Exercise ST Segment (slope)", options=["Upsloping", "Flat", "Downsloping"], index=0)
                
                ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', options=[0, 1, 2, 3, 4], index=0)
                
                thal = st.selectbox('Status of the Heart (thal)', options=["Normal", "Fixed Defect", "Reversible Defect"], index=0)


            # prediciton button
            predict_btn = st.form_submit_button("Predict❤️", use_container_width=True)
    
    # main column 2
    with main_col2:
        
        # when prediction button is clicked
        if predict_btn:
            
            # define numeric variables
            patient_age = age
            patient_trestbps = trestbps
            patient_chol = chol
            patient_thalach = thalach
            patient_oldpeak= oldpeak
            
            
            # Perform manual data encoding for categorical variables
            
            patient_sex = 0 # Female
            if sex == "Male":
                patient_sex = 1
                
            patient_cp = [0, 0, 0]   # Typical Angina
            if cp == "Atypical Angina":
                patient_cp = [1, 0, 0]
            elif cp == "Non-Anginal Pain":
                patient_cp = [0, 1, 0]
            elif cp == "Asymptomatic":
                patient_cp = [0, 0, 1]
                
            patient_fbs = 1
            if fbs == "Less Than 120 mg/dl":
                patient_fbs = 0
                
            patient_restecg = [0, 0]    # Normal
            if restecg == "Abnormal":
                patient_restecg = [1, 0]
            elif restecg == "Ventricular Hypertrophy":
                patient_restecg = [0, 1]
                
            patient_exang = 0   # No
            if patient_exang == "Yes":
                patient_exang = 1
                
            patient_slope = 0    # Upsloping
            if slope == "Flat":
                patient_slope = 1
            elif slope == "Downsloping":
                patient_slope = 2
                
            patient_ca = ca
                
            patient_thal = [0, 0]    # Normal
            if slope == "Fixed Defect":
                patient_slope = [1, 0]
            elif slope == "Reversible Defect":
                patient_slope = [0, 1]
            
            
            # define data list to store patient data
            data = [patient_age, patient_sex, patient_trestbps, patient_chol, patient_fbs, patient_thalach, patient_exang, patient_oldpeak, patient_slope, patient_ca]
            data.extend(patient_cp) # extend list
            data.extend(patient_restecg)    # extend list
            data.extend(patient_thal)   # extend list
            
            # define column names
            coulmn_names = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'cp_1', 'cp_2', 'cp_3', 'restecg_1', 'restecg_2', 'thal_2', 'thal_3']
            
            # create a dataframe from user input (data list)
            # note that data is already encoded with the above if statements
            df_encoded = pd.DataFrame([data], columns=coulmn_names)
            
        # If manual data encoding was not completed, the following code could have been executed. (note that dummy_columns.pkl would have to be saved to disk in the model python file first)
        # # Apply categorical encoding
        # input_df = pd.get_dummies(input_df)

        # # Load the saved dummy columns and lambdas
        # dummy_columns = joblib.load('dummy_columns.pkl')
        # # Ensure the dummy columns match those from training
        # input_df = input_df.reindex(columns=dummy_columns, fill_value=0)
            
            # load saved lambdas
            lambdas = joblib.load('boxcox_lambdas.pkl')

            # Transform input using the saved Box-Cox lambdas
            input_data_df = transform_input(df_encoded, lambdas)
            
            # call predict from prediction.py file on the dataframe
            prediction = predict(df_encoded)
            
            # Check prediction value and output relevant message
            if prediction[0] == 1:  # patient has heart disease
                st.header(":violet[Prediction is:]", divider='red')
                st.header(f":red[Patient has heart disease ❤️‍🩹😢]", divider='red')
            else:   # patients does not have heart disease
                st.header(":violet[Prediction is:]", divider='green')
                st.header(f":green[Patient does not have heart disease ❤️😁] ", divider='green')
                
