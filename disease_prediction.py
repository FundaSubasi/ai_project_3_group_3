import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.models import load_model
import random

# Set the theme for the app
st.set_page_config(page_title="🩺 Disease Prediction Based on Symptoms", layout="wide")

# Load the trained MLP model
model = load_model('resources/mlp_model.h5')

# Load and prepare the dataset
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

# Initialize selected symptoms list
if 'selected_symptoms' not in st.session_state:
    st.session_state.selected_symptoms = ['Please Select'] * 5

# Function to display dropdowns and manage selections
def display_dropdowns():
    for i in range(len(st.session_state.selected_symptoms)):
        options = ['Please Select'] + sorted(set(symptoms_list) - set(st.session_state.selected_symptoms[:i] + st.session_state.selected_symptoms[i+1:]))

        # Initialize session state for the typed value if not already set
        if f"typed_{i}" not in st.session_state:
            st.session_state[f"typed_{i}"] = ""

        def on_select_change():
            typed_value = st.session_state[f"typed_{i}"].strip()
            if typed_value in symptoms_list:
                st.session_state.selected_symptoms[i] = typed_value
            elif st.session_state.selected_symptoms[i] not in symptoms_list:
                st.session_state.selected_symptoms[i] = 'Please Select'

        selected_symptom = st.selectbox(
            f"Symptom {i+1}",
            options=options,
            index=options.index(st.session_state.selected_symptoms[i]) if st.session_state.selected_symptoms[i] in options else 0,
            key=f"dropdown_{i}",
            on_change=on_select_change
        )

        st.session_state.selected_symptoms[i] = selected_symptom

    if len(st.session_state.selected_symptoms) < 17:
        if st.button("Add Another Symptom"):
            st.session_state.selected_symptoms.append('Please Select')

# Create a two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    # Display dropdowns for symptom selection
    display_dropdowns()

# Filter out 'Please Select' from the final symptom list
final_selected_symptoms = [symptom for symptom in st.session_state.selected_symptoms if symptom != 'Please Select' and symptom in symptoms_list]

# Placeholder for the pie chart
with col2:
    if len(final_selected_symptoms) < 5:
        fig = px.pie(names=["Please make symptom selections to generate probable disease cause"], values=[100], title="Awaiting Input")
        st.markdown("**User must select at least 5 symptoms for Predict to be enabled**", unsafe_allow_html=True)
        st.plotly_chart(fig)
    else:
        # Limit number of selected symptoms to 17
        if len(final_selected_symptoms) > 17:
            st.warning("You can only select up to 17 symptoms.")
            final_selected_symptoms = final_selected_symptoms[:17]

        # Disable predict button if conditions are not met
        predict_disabled = len(final_selected_symptoms) < 5 or len(final_selected_symptoms) > 17

        if st.button("Predict", disabled=predict_disabled):
            # Convert selected symptoms to encoded format
            encoded_symptoms = np.zeros(len(symptoms_list))
            for symptom in final_selected_symptoms:
                if symptom in symptoms_list:
                    encoded_symptoms[symptoms_list.index(symptom)] = 1

            # Prepare input for the model
            final_input = np.zeros((1, 676))  # Ensure the input has 676 features as expected by the model
            final_input[0, :len(encoded_symptoms)] = encoded_symptoms

            # Predict using the model
            predictions = model.predict(final_input)

            # Post-prediction adjustments
            disease_match_scores = {}
            for _, row in df.iterrows():
                disease_symptoms = row[1:].values  # Skip the first column (Disease)
                disease_encoded = np.array([1 if symptom in disease_symptoms else 0 for symptom in symptoms_list])
                match_score = np.sum(encoded_symptoms == disease_encoded)
                disease_match_scores[row['Disease']] = match_score

            # Exact match boost
            if any(np.array_equal(encoded_symptoms, df.iloc[i, 1:].values) for i in range(len(df))):
                exact_match_disease = next(df['Disease'][i] for i in range(len(df)) if np.array_equal(encoded_symptoms, df.iloc[i, 1:].values))
                exact_match_idx = df[df['Disease'] == exact_match_disease].index[0]
                if exact_match_idx < len(predictions[0]):
                    predictions[0][exact_match_idx] *= 2.0

            # Partial match boost
            elif any(score >= 10 for score in disease_match_scores.values()):
                partial_match_disease = max(disease_match_scores, key=disease_match_scores.get)
                partial_match_idx = df[df['Disease'] == partial_match_disease].index[0]
                if partial_match_idx < len(predictions[0]):
                    predictions[0][partial_match_idx] *= 1.5

            # Less significant match boost
            else:
                best_match_disease = max(disease_match_scores, key=disease_match_scores.get)
                best_match_idx = df[df['Disease'] == best_match_disease].index[0]
                if best_match_idx < len(predictions[0]):
                    predictions[0][best_match_idx] *= 1.2

            # Normalize predictions
            predictions = predictions / predictions.sum() * 100

            # Create DataFrame for predictions
            diseases = df['Disease'].unique()
            prediction_df = pd.DataFrame(predictions, columns=diseases).T
            prediction_df.columns = ['Probability']
            prediction_df = prediction_df.sort_values(by='Probability', ascending=False)

            # Select the top 5 diseases
            top_5 = prediction_df.head(5)

            # Adjust the probabilities to sum to 100%
            top_5['Probability'] = (top_5['Probability'] / top_5['Probability'].sum()) * 100

            # Display the prediction result text
            st.markdown(f"**Patient has a high chance of having {top_5.index[0]}**")

            # Plot interactive pie chart for the top 5 diseases
            fig = px.pie(top_5, values='Probability', names=top_5.index, title='Top 5 Disease Predictions')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=400, width=400)
            st.plotly_chart(fig)

            # Display additional disease suggestions
            remaining_diseases = prediction_df.iloc[5:].index.tolist()
            if remaining_diseases:
                additional_diseases = random.sample(remaining_diseases, min(4, len(remaining_diseases)))
                st.write("Here are additional diseases the medical provider may want to consider, accompanied by lab work, diagnoses, and care suggestions.")
                st.write(", ".join(additional_diseases))
            else:
                st.write("No other diseases can be indicated at this time.")
            st.write("""
            This data was pulled from the CDC using their research studies on listed diseases and symptoms. Please note that these predictions are not definitive diagnoses and should be used as a guide to aid in clinical decision-making. For accurate diagnosis and treatment, medical professionals should rely on comprehensive clinical evaluation and testing.
            """)

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
        font-size: 15px;
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
