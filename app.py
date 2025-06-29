import streamlit as st
import joblib 
import numpy as np

#lets load our model
model = joblib.load('model/titanic_final_model.pkl')

st.title("🚢 Titanic Survival Predictor")

#Taking inputs from user

pclass = st.selectbox("Passenger Class",[1, 2, 3])
sex = st.selectbox("Sex",['Male','Female'])
age = st.slider("Age",0, 100, 25)
fare = st.slider("Fare", 0.0, 300.00,50.00)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 5, 0)
parch = st.slider("Parents/Children Aboard", 0, 5, 0)
embarked = st.selectbox("Port of Embarkation", ['Q', 'S'])

#convert to numeric inputs (match our training encoding)
sex_encoded = 1 if sex == 'Male' else 0
embarked_S = 1 if embarked == 'S' else 0
embarked_Q = 1 if embarked == 'Q' else 0
# Dummy PassengerId (since it’s not relevant, but required by model shape)
passenger_id = 0

#prepare input
features =np.array([[passenger_id, pclass, sex_encoded, age, sibsp, parch, fare, embarked_Q, embarked_S]])

#prediction
prediction = model.predict(features)

#show result
st.markdown(f"### 🎯 Predicted: {'Survived ✅' if prediction[0] == 1 else 'Did Not Survive ❌'}")
st.write("Input Features:", features)
