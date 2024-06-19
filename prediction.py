import joblib

# Predict the outcome
def predict(df_encoded):
    # Convert DataFrame to the format expected by the model
    data = df_encoded.values

    # Load the trained model
    clf = joblib.load("svm_heart_disease_model.sav")

    # Make predictions
    return clf.predict(data)