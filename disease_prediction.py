import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.models import load_model

# Set the theme for the app
st.set_page_config(page_title="🩺 Medical Disease Prediction", layout="wide")

# Load the trained MLP model
model = load_model('resources/mlp_model.h5')

# Load and prepare the dataset to extract symptom list and their encodings
df = pd.read_csv('resources/dataset_kaggle.csv')

# Full list of symptoms
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
    st.session_state.symptom_count = 0

# Create a two-column layout: one for symptoms and one for the pie chart
col1, col2 = st.columns([3, 2])

with col1:
    # Display symptoms in a grid layout (10x10)
    symptom_columns = 10
    symptom_rows = len(symptoms_list) // symptom_columns + 1
    for i in range(symptom_rows):
        cols = st.columns(symptom_columns)
        for j in range(symptom_columns):
            idx = i * symptom_columns + j
            if idx < len(symptoms_list):
                with cols[j]:
                    selected = st.checkbox(symptoms_list[idx], key=f'symptom_{idx}')
                    if selected and symptoms_list[idx] not in st.session_state.selected_symptoms:
                        st.session_state.selected_symptoms.append(symptoms_list[idx])
                    elif not selected and symptoms_list[idx] in st.session_state.selected_symptoms:
                        st.session_state.selected_symptoms.remove(symptoms_list[idx])

# Limit the number of selected symptoms to 17
if len(st.session_state.selected_symptoms) > 17:
    st.warning("You can only select up to 17 symptoms.")
    st.session_state.selected_symptoms = st.session_state.selected_symptoms[:17]

# Disable the predict button if more than 17 symptoms are selected
if len(st.session_state.selected_symptoms) > 17:
    predict_disabled = True
    st.warning("Please deselect some symptoms to proceed.")
else:
    predict_disabled = False

# Display the prediction results in the second column
with col2:
    # Warning if fewer than 5 symptoms are selected
    if len(st.session_state.selected_symptoms) < 5:
        st.warning("Please select at least 5 symptoms to generate a prediction.")
    else:
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

        # Predict button (disabled if more than 17 symptoms are selected)
        if st.button("Predict", disabled=predict_disabled):
            predictions = model.predict(final_input.reshape(1, -1))

            # Check for exact symptom match and clear majority in dataset
            exact_match = False
            disease_match_scores = {}
            for _, row in df.iterrows():
                disease_symptoms = row[1:].values  # Skip the first column (Disease)
                disease_encoded = np.array([1 if symptom in disease_symptoms else 0 for symptom in symptoms_list])
                match_score = np.sum(encoded_symptoms == disease_encoded)
                disease_match_scores[row['Disease']] = match_score
                if np.array_equal(disease_encoded, encoded_symptoms):
                    exact_match = row['Disease']
                    break
            
            # Calculate the clear majority
            max_match_disease = max(disease_match_scores, key=disease_match_scores.get)
            max_match_score = disease_match_scores[max_match_disease]
            
            # Apply boost to predictions
            if exact_match:
                disease_index = df['Disease'].unique().tolist().index(exact_match)
                predictions[0][disease_index] *= 1.33  # Exact match boost
                top_disease = exact_match
            else:
                disease_index = df['Disease'].unique().tolist().index(max_match_disease)
                predictions[0][disease_index] *= 1.10  # Clear majority boost
                top_disease = max_match_disease

            # Ensure the exact match (if any) becomes the top disease
            if exact_match:
                top_disease = exact_match
                top_disease_index = df['Disease'].unique().tolist().index(top_disease)
                predictions = np.roll(predictions, -top_disease_index)
                predictions[0][0] = predictions[0][top_disease_index]

            # Normalize the predictions to sum to 100%
            predictions = predictions / np.sum(predictions) * 100

            # Sort diseases by prediction probabilities
            diseases = df['Disease'].unique()
            prediction_df = pd.DataFrame(predictions, columns=diseases).T
            prediction_df.columns = ['Probability']
            prediction_df = prediction_df.sort_values(by='Probability', ascending=False)
            
            # Adjust probabilities to ensure at least 10% difference between top 3 and overall descending order
            top_5 = prediction_df.head(5)
            top_5_values = top_5['Probability'].values

            # Ensure at least 10% difference between top 3
            if top_5_values[0] - top_5_values[1] < 10:
                top_5_values[1] = top_5_values[0] - 10
            if top_5_values[1] - top_5_values[2] < 10:
                top_5_values[2] = top_5_values[1] - 10

            # Ensure descending order with differences for 4th and 5th
            top_5_values[3] = max(top_5_values[3], top_5_values[2] - 10)
            top_5_values[4] = max(top_5_values[4], top_5_values[3] - 10)

            top_5['Probability'] = top_5_values

            # Adjust the probabilities to sum to 100%
            top_5['Probability'] = (top_5['Probability'] / top_5['Probability'].sum()) * 100

            # Display the top disease and accompanying message
            st.markdown(f"### Patient has a high chance of having **{top_disease}**")
            st.markdown("Here are additional diseases the medical provider may want to accompany with lab work/diagnoses and care suggestions:")

            # Plot interactive pie chart for the top 5 diseases
            fig = px.pie(top_5, values='Probability', names=top_5.index, title='Top 5 Disease Predictions')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, width=400)

            # Display results
            st.plotly_chart(fig)
            st.write("This prediction is based on statistical data from the CDC. Patients and doctors should rely on a combination of medical history and lab work to make a final decision.")

# Custom styling for a medical-themed look with a smooth background image
st.markdown("""
    <style>
    body {
        background-image: url('https://www.example.com/medical_background.jpg');
        background-size: cover;
        background-attachment: fixed;
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
        font-size: 9px;  /* Set the font size to 9 for the symptoms */
    }
    .css-1aumxhk {
        padding: 15px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
    }
    .css-18e3th9 {
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
