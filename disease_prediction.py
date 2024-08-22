import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.models import load_model

# Set the theme for the app
st.set_page_config(page_title="Medical Disease Prediction", layout="wide")

# Load the trained MLP model
model = load_model('resources/mlp_model.h5')

# Load and prepare the dataset to extract symptom list and their encodings
df = pd.read_csv('resources/dataset_kaggle.csv')

# Extract symptoms list
symptoms_list = ['Anemia', 'Anxiety', 'Aura', 'Belching', 'Bladder issues', 'Bleeding mole', 
                 'Blisters', 'Bloating', 'Blood in stool', 'Body aches', 'Bone fractures', 
                 'Bone pain', 'Bowel issues', 'Burning', 'Butterfly-shaped rash', 
                 'Change in bowel habits', 'Change in existing mole', 'Chest discomfort', 
                 'Chest pain', 'Congestion', 'Constipation', 'Coughing up blood', 'Depression', 
                 'Diarrhea', 'Difficulty performing familiar tasks', 'Difficulty sleeping', 
                 'Difficulty swallowing', 'Difficulty thinking', 'Difficulty walking', 
                 'Double vision', 'Easy bruising', 'Fatigue', 'Fear', 'Frequent infections', 
                 'Frequent urination', 'Fullness', 'Gas', 'Hair loss', 'Hard lumps', 'Headache', 
                 'Hunger', 'Inability to defecate', 'Increased mucus production', 
                 'Increased thirst', 'Irregular heartbeat', 'Irritability', 'Itching', 
                 'Jaw pain', 'Limited range of motion', 'Loss of automatic movements', 
                 'Loss of height', 'Loss of smell', 'Loss of taste', 'Lump or swelling', 
                 'Mild fever', 'Misplacing things', 'Morning stiffness', 'Mouth sores', 
                 'Mucus production', 'Nausea', 'Neck stiffness', 'Nosebleeds', 'Numbness', 
                 'Pain during urination', 'Pale skin', 'Persistent cough', 'Persistent pain', 
                 'Pigment spread', 'Pneumonia', 'Poor judgment', 'Problems with words', 
                 'Rapid pulse', 'Rash', 'Receding gums', 'Redness', 'Redness in joints', 
                 'Reduced appetite', 'Seizures', 'Sensitivity to light', 'Severe headache', 
                 'Shortness of breath', 'Skin changes', 'Skin infections', 'Slight fever', 
                 'Sneezing', 'Sore that doesn’t heal', 'Soreness', 'Staring spells', 
                 'Stiff joints', 'Stooped posture', 'Swelling', 'Swelling in ankles', 
                 'Swollen joints', 'Swollen lymph nodes', 'Tender abdomen', 'Tenderness', 
                 'Thickened skin', 'Throbbing pain', 'Tophi', 'Tremor', 'Unconsciousness', 
                 'Unexplained bleeding', 'Unexplained fevers', 'Vomiting', 'Weakness', 
                 'Withdrawal from work', 'Writing changes']

# Streamlit app layout
st.title("🩺 Disease Prediction Based on Symptoms")
st.markdown("""
Welcome to the Disease Prediction app. This tool allows healthcare providers and patients to input symptoms and receive potential disease predictions based on machine learning. The predictions prioritize serious illnesses depending on the symptoms provided.
""")

# Initialize the selected symptoms list and state variables
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = []
if 'symptom_count' not in st.session_state:
    st.session_state.symptom_count = 1

# Function to add a symptom dropdown
def add_symptom():
    if st.session_state.symptom_count < 17:
        st.session_state.symptom_count += 1

# Function to remove a symptom
def remove_symptom(symptom):
    if symptom in st.session_state.selected_symptoms:
        st.session_state.selected_symptoms.remove(symptom)

# Display symptom dropdowns
for i in range(st.session_state.symptom_count):
    symptom = st.selectbox(f'Select Symptom {i+1}', ['None'] + symptoms_list, key=f'symptom_{i+1}')
    if symptom != 'None' and symptom not in st.session_state.selected_symptoms:
        st.session_state.selected_symptoms.append(symptom)

# Button to add more symptoms
if st.session_state.symptom_count < 17:
    st.button("Add Symptom", on_click=add_symptom)

# Display selected symptoms with a small 'x' to remove in a 6x3 grid layout
st.write("### Selected Symptoms:")
columns = st.columns(6)
for idx, symptom in enumerate(st.session_state.selected_symptoms):
    with columns[idx % 6]:
        st.write(f"{symptom} ", st.button("❌", key=f'remove_{symptom}', on_click=remove_symptom, args=(symptom,)))

# Convert selected symptoms to encoded format
encoded_symptoms = np.zeros(len(symptoms_list))
for symptom in st.session_state.selected_symptoms:
    if symptom in symptoms_list:
        encoded_symptoms[symptoms_list.index(symptom)] = 1

# Logic-based Symptom Weighting (Simple Math-based Approach)
symptom_weights = np.array([2.0 if symptom in ['Chest pain', 'Severe headache', 'Shortness of breath', 'Coughing up blood'] else 1.0 for symptom in symptoms_list])

# Calculate the weighted input
weighted_input = encoded_symptoms * symptom_weights

# Prepare final input for the model (match model's expected input shape)
final_input = np.zeros(676)
final_input[:len(weighted_input)] = weighted_input

# Predict button (only enabled if at least one symptom is selected)
if len(st.session_state.selected_symptoms) > 0:
    if st.button("Predict"):
        predictions = model.predict(final_input.reshape(1, -1))

        # Basic logic: if fewer symptoms and none severe, prioritize less serious conditions
        serious_threshold = 60.0
        if np.sum(weighted_input) > serious_threshold:
            predictions = predictions * 1.5  # Increase probability for serious conditions
        else:
            predictions = predictions * 0.5  # Decrease probability for serious conditions

        # Sort diseases by prediction probabilities
        diseases = df['Disease'].unique()
        prediction_df = pd.DataFrame(predictions, columns=diseases).T
        prediction_df.columns = ['Probability']
        prediction_df = prediction_df.sort_values(by='Probability', ascending=False)
        
        # Select the top 5 diseases
        top_5 = prediction_df.head(5)
        
        # Adjust the probabilities to sum to 100%
        top_5['Probability'] = (top_5['Probability'] / top_5['Probability'].sum()) * 100

        # Plot interactive pie chart for the top 5 diseases
        fig = px.pie(top_5, values='Probability', names=top_5.index, title='Top 5 Disease Predictions')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=500, width=800)

        # Display results
        st.plotly_chart(fig)
        st.write("This prediction is based on statistical data from the CDC. Patients and doctors should rely on a combination of medical history and lab work to make a final decision.")
else:
    st.write("Please select at least one symptom to generate a prediction.")

# Custom styling for a medical-themed look
st.markdown("""
    <style>
    body {
        background-color: #f0f5f9;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px;
        margin-top: 8px;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMarkdown {
        font-family: Arial, sans-serif;
        color: #333333;
        font-size: 15px;
    }
    .css-1aumxhk {
        padding: 15px;
        background: #ffffff;
        border-radius: 10px;
    }
    .css-18e3th9 {
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
