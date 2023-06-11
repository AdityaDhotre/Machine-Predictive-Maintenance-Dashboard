import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore')
import lightgbm as lgb

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('predictive_maintenance/predictive_maintenance.csv')


# Preprocess the data
def preprocess_data(df):
    train, test = train_test_split(df, test_size=0.20, random_state=1)

    train = train.drop(['FailureType'], axis=1)
    test = test.drop(['FailureType'], axis=1)

    train['Type'] = train['Type'].replace(['H', 'L', 'M'], [1, 2, 3])
    test['Type'] = test['Type'].replace(['H', 'L', 'M'], [1, 2, 3])

    train = train.drop(['ProductID'], axis=1)
    test = test.drop(['ProductID'], axis=1)

    train = train.drop(['UDI'], axis=1)
    test = test.drop(['UDI'], axis=1)

    # Remove special characters from column names
    train.columns = train.columns.str.replace('[^a-zA-Z0-9]', '_')
    test.columns = test.columns.str.replace('[^a-zA-Z0-9]', '_')

    return train, test


# Train the model
def train_model(train, X_train, y_train):
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    return model


# Predict maintenance requirements
def predict_maintenance(model, data):
    # Perform prediction using the trained model
    predictions = model.predict(data)

    return predictions


# Streamlit app
def main():
    st.title('Machine Predictive Maintenance')

    # Load the data
    data = load_data()

    # Preprocess the data
    train, test = preprocess_data(data)

    y = train['Target']
    X = train.drop(['Target'], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)

    # Train the model
    model = train_model(train, X_train, y_train)
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)

     # Get user inputs
    Type = st.selectbox('Type', ['H', 'L', 'M'])
    AirTemp = st.number_input('Air Temperature', value=0.0)
    ProcessTemp = st.number_input('Process Temperature', value=0.0)
    RotationalSpeed = st.number_input('Rotational Speed', value=0.0)
    Torque = st.number_input('Torque', value=0.0)
    ToolWear = st.number_input('Tool Wear', value=0.0)

    # Convert Type to integer
    if Type == 'H':
        Type = 1
    elif Type == 'L':
        Type = 2
    elif Type == 'M':
        Type = 3

    # Create DataFrame for user input
    new_Data = pd.DataFrame({
        'Type': [Type],
        'Air_Temperature_K': [AirTemp],
        'Process_Temperature_K': [ProcessTemp],
        'Rotational_speed_rpm': [RotationalSpeed],
        'Torque_Nm': [Torque],
        'Tool_wear_min': [ToolWear],
    })

    # Make predictions for the new data
    predictions = predict_maintenance(model, new_Data)
    
    # Display the predictions
    st.subheader('Predictions')
    if predictions[0] == 1:
        st.write('Maintenance is required.')
    else:
        st.write('Maintenance is not required.')


    # Display the accuracy
    st.subheader('Accuracy')
    st.text(f"Accuracy: {accuracy*100}% ")


if __name__ == '__main__':
    main()
