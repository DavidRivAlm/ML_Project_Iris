import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_scaled, y_train)

# Save the trained model and scaler as pickle files
with open('logistic_regression_model_iris.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)

with open('scaler_iris.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Streamlit app code to make predictions and display results
def make_predictions(input_data):
    # Load the model and scaler
    with open('logistic_regression_model_iris.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('scaler_iris.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Scale the input data
    input_data_scaled = scaler.transform([input_data])

    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_class = target_names[prediction][0]
    
    return predicted_class

# Streamlit app UI
st.title("Iris Flower Species Prediction")
st.write("This app uses a logistic regression model to predict the species of the Iris flower based on four features.")

# Get user input for the four features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Prepare the input data
input_data = [sepal_length, sepal_width, petal_length, petal_width]

# Show prediction when the user clicks the button
if st.button('Predict Species'):
    predicted_class = make_predictions(input_data)
    st.write(f"The predicted species is: {predicted_class}")
    
    # Display the model accuracy on test set
    y_pred = lr.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy on test data: {accuracy:.2f}")


