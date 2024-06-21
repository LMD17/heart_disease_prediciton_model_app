# ITDAA4-12 - Heart Prediciton Streamlit App

Streamlit Web App link: https://heartdiseasepredicitonmodelapp-dvdsolyraihyydknwxqmm3.streamlit.app/

## Introduction
The webapp provides a model to determine if a patient has heart disease. The user (doctor) can input the patients information and the SVM model predict whether the patient has heart disease and needs treatement, or the patient does not have heart disease.

## Prerequisites
Python: Ensure Python is installed on your system. You can download it from python.org.
Pip: Python package installer. It comes with Python by default.


## Installation
1. Install required libraries:
    ```
    pip install streamlit joblib scikit-learn sklearn scipy seaborn numpy pandas
    ```

## Run model.py
```
    python model.py
```
This will perform
- data cleaning
- feature comparison and analysis
- data preprocessing
- feature encoding
- feature transformation
- model building
- model evaluation 
- saving model to disk

## Run app.py
```
    streamlit run app.py   
```
This will run the Streamlit app and allow the user to make heart disease predicitons.