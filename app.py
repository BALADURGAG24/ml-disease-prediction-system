from flask import Flask, render_template, request
import joblib
import numpy as np
from collections import Counter  # Use this instead of scipy.stats.mode

# Load models and encoder
rf_model = joblib.load('models/rf_model.pkl')
nb_model = joblib.load('models/nb_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
encoder = joblib.load('models/label_encoder.pkl')
symptoms = joblib.load('models/symptoms_list.pkl')

symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

# Create Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    input_symptoms = request.form.getlist('symptoms')

    input_data = [0] * len(symptom_index)
    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_pred = encoder.classes_[rf_model.predict(input_data)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_data)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_data)[0]]

    # Use Counter to get the most common prediction (final prediction)
    predictions = [rf_pred, nb_pred, svm_pred]
    final_pred = Counter(predictions).most_common(1)[0][0]

    return render_template(
        'result.html',
        rf_pred=rf_pred,
        nb_pred=nb_pred,
        svm_pred=svm_pred,
        final_pred=final_pred
    )

if __name__ == '__main__':
    app.run(debug=True)
