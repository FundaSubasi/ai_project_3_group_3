# Disease Predictor AI Models and Web App
# Overview 
This project aims to predict potential diseases based on symptoms provided by the user, leveraging a comprehensive dataset of 41 diseases and approximately 60 varying symptoms. We have developed a predictive model using advanced neural network architectures—Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM) and a transformer model (DistilBERT)—to accurately identify diseases from given symptoms.

# Models
- ***Multilayer Perceptron (MLP):*** A feedforward neural network that maps sets of input data onto a set of appropriate outputs. MLP was chosen for its robustness in handling tabular data. The MLP model was integrated into the Streamlit application for real-time disease prediction.
- ***Convolutional Neural Network (CNN):*** Though predominantly used for image data, CNNs have been adapted in this project to treat symptom data as one-dimensional spatial vectors, extracting patterns that are essential for disease prediction.
- ***Long Short-Term Memory (LSTM):*** A type of recurrent neural network (RNN) useful in sequence prediction problems. LSTM was utilized considering the sequence in which symptoms are reported could provide significant predictive power.
- ***Transformer Model (DistilBERT):*** We also experimented with the distilbert-base-uncased transformer model, trained using Hugging Face over approximately 5 hours. However, this model performed poorly with an evaluation loss of 2.38, which was significantly higher compared to the other models.

## Neural Network Model Performance
Each of the neural network models—MLP, CNN, and LSTM—achieved zero loss and 100% accuracy after only a few epochs of training. The simplicity of the dataset, which is non-temporal and essentially one-dimensional in nature, contributed significantly to the ease with which the models could learn and make predictions. This performance suggests that the models could effectively capture and utilize the patterns within the dataset without the complexities often associated with time-series or multi-dimensional data.

# Dataset
The dataset for this project was sourced from Kaggle and consists of two main components:

**Symptom-Disease DataFrame:** This primary dataset includes 41 diseases and around 60 symptoms, where each entry lists a disease along with its corresponding symptoms.

**Disease Weights DataFrame:** An additional dataset provided in the repository assigns weights to each symptom on a scale from 1 to 7, with higher weights indicating more severe symptoms. For example, 'itching' is assigned a weight of 1, while 'stomach pain' is assigned a weight of 5. These weights were integrated into the main dataset by calculating a weighted sum for each disease based on the severity of its associated symptoms. This weighted sum was then added to the original dataframe, enhancing the training data to include disease, symptoms, and their respective weights.


# Tools and Technologies
- Python: The primary programming language used.
- TensorFlow/Keras: For building and training the neural network models.
- Streamlit: To create a user-friendly web application that interacts with the trained models.
- Pandas/Numpy: For data manipulation and numerical operations.
- Hugging Face: For training the DistilBERT transformer model.

# Streamlit Application
The Streamlit application serves as the interface for users to interact with the MLP model. Users can input their symptoms, and the application will predict potential diseases. The application is designed to be intuitive and accessible for non-technical users.

# Usage
After starting the Streamlit application, follow the on-screen instructions to input symptoms and receive disease predictions.

# Contributing
This project was created by Priscilla Morales, J'Mari Hawkins, Andy Bhanderi, Funda Subasi, Kyle Prudente and Peta-Gaye Mckenzie
