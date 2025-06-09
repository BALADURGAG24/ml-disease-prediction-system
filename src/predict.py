# src/predict.py

import numpy as np
import joblib
from scipy.stats import mode

# Load models and encoder
rf_model = joblib.load('../models/rf_model.pkl')
nb_model = joblib.load('../models/nb_model.pkl')
svm_model = joblib.load('../models/svm_model.pkl')
encoder = joblib.load('../models/label_encoder.pkl')
symptoms = joblib.load('../models/symptoms_list.pkl')

symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])[0][0]

    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

if __name__ == "__main__":
    print(predict_disease("Itching,Skin Rash,Nodal Skin Eruptions"))
