# AI Pathos-Predictor: NN/NLP Models and Web App
# Overview 
This project aims to predict potential diseases based on symptoms provided by the user, leveraging a comprehensive dataset of 40 diseases and approximately 60 varying symptoms. We have developed a predictive model using advanced neural network architectures—Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM) and a transformer model (DistilBERT)—to accurately identify diseases from given symptoms.

# Models
- Multilayer Perceptron (MLP): A feedforward neural network that maps sets of input data onto a set of appropriate outputs. MLP was chosen for its robustness in handling tabular data. The MLP model was integrated into the Streamlit application for real-time disease prediction.
- Convolutional Neural Network (CNN): Though predominantly used for image data, CNNs have been adapted in this project to treat symptom data as one-dimensional spatial vectors, extracting patterns that are essential for disease prediction.
- Long Short-Term Memory (LSTM): A type of recurrent neural network (RNN) useful in sequence prediction problems. LSTM was utilized considering the sequence in which symptoms are reported could provide significant predictive power.
- Transformer Model (DistilBERT): We also tried using a tranformer model since the dataset was in natural language. We chose DistilBERT for its strong natural language processing abilities. We presumed this model could effectively understand and classify the textual data, leading to accurate disease predictions.

## Model Performance
- **Multi-Layer Perceptron (MLP):**  
  The MLP model showed strong performance, with a training accuracy of 100% and a test accuracy of 81%. This indicates that the model was able to learn the patterns in the training data effectively and generalize relatively well to unseen data. The model's loss was also low, making it the best performer among the models tested.

- **Convolutional Neural Network (CNN):**  
  The CNN model performed well with a test accuracy of approximately 80.5%. Although the training accuracy was high, the model experienced a higher loss during testing, suggesting that while it captured some patterns in the data, it wasn't as effective as the MLP in generalizing to new examples.

- **Recurrent Neural Network (RNN) with LSTM:**  
  The LSTM model was modified several times to improve its performance. Despite these modifications, the LSTM performed poorly, with a test accuracy of only 2.25% and a high test loss. This suggests that the LSTM struggled to learn meaningful patterns from the data, likely due to the nature of the dataset, which may not be well-suited for a sequential model like LSTM.

- **Transformer Model (DistilBERT):** We experimented with the distilbert-base-uncased transformer model, trained using Hugging Face over approximately 5 hours. However, this model performed poorly with an evaluation loss of 2.38, which was higher than the neural network models.

### Conclusion:
Based on the results, the **Multi-Layer Perceptron (MLP)** model performed the best, showing strong accuracy and low loss. The **Convolutional Neural Network (CNN)** also performed reasonably well but had a slightly higher test loss. The **Recurrent Neural Network (RNN) with LSTM** and the **Transformer Model (DistilBERT):**, despite several modifications, did not perform well, indicating that it might not be the right choice for this dataset.

# Dataset
The dataset for this project was sourced from Kaggle ([dataset link](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset/discussion/529794)) and those data were sourced from the CDC.

**Symptom-Disease DataFrame Example:** diseases_data = 
"Common Cold": ["Runny nose", "Sore throat", "Cough", "Fever", "Fatigue", "Sneezing", "Congestion", "Headache", "Mild fever", "Body aches"],

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
