# disease_dataset_generator.py

import pandas as pd
import numpy as np
import random

# Define diseases and symptom probabilities (tune as needed!)
disease_profiles = {
    'Common Cold': {
        'Runny Nose': 0.9,
        'Cough': 0.7,
        'Sore Throat': 0.8,
        'Fever': 0.3,
        'Fatigue': 0.5
    },
    'Flu': {
        'Fever': 0.95,
        'Cough': 0.8,
        'Sore Throat': 0.6,
        'Fatigue': 0.9,
        'Body Aches': 0.85,
        'Headache': 0.7
    },
    'COVID-19': {
        'Fever': 0.9,
        'Cough': 0.85,
        'Shortness of Breath': 0.7,
        'Loss of Smell': 0.6,
        'Fatigue': 0.75,
        'Headache': 0.5
    },
    'Gastroenteritis': {
        'Vomiting': 0.9,
        'Diarrhea': 0.95,
        'Abdominal Pain': 0.8,
        'Fever': 0.4,
        'Fatigue': 0.6
    },
    'Migraine': {
        'Headache': 0.95,
        'Nausea': 0.8,
        'Blurred Vision': 0.5,
        'Sensitivity to Light': 0.85,
        'Fatigue': 0.5
    },
    'Hypertension': {
        'Headache': 0.5,
        'Dizziness': 0.6,
        'Blurred Vision': 0.4,
        'Fatigue': 0.3,
        'Chest Pain': 0.2
    },
    'Diabetes': {
        'Increased Thirst': 0.9,
        'Frequent Urination': 0.85,
        'Fatigue': 0.7,
        'Blurred Vision': 0.5,
        'Slow Healing Wounds': 0.6
    },
    'Pneumonia': {
        'Fever': 0.9,
        'Cough': 0.9,
        'Chest Pain': 0.7,
        'Shortness of Breath': 0.8,
        'Fatigue': 0.7
    },
    'Asthma': {
        'Cough': 0.7,
        'Shortness of Breath': 0.9,
        'Chest Tightness': 0.8,
        'Wheezing': 0.85,
        'Fatigue': 0.6
    },
    'Anemia': {
        'Fatigue': 0.95,
        'Pale Skin': 0.8,
        'Shortness of Breath': 0.6,
        'Headache': 0.5,
        'Dizziness': 0.7
    }
}

# Build global symptom list
all_symptoms = sorted({symptom for profile in disease_profiles.values() for symptom in profile})

# Parameters
n_samples = 10000
noise_prob = 0.02  # noise probability for non-relevant symptoms

# Generate data
data_rows = []

for _ in range(n_samples):
    # Pick a random disease
    disease = random.choice(list(disease_profiles.keys()))
    profile = disease_profiles[disease]
    
    row = {}
    for symptom in all_symptoms:
        # Use disease-specific probability or small noise for irrelevant symptoms
        if symptom in profile:
            prob = profile[symptom]
        else:
            prob = noise_prob
        
        row[symptom] = int(np.random.rand() < prob)
    
    # Assign the target label
    row['Disease'] = disease
    data_rows.append(row)

# Create DataFrame
df = pd.DataFrame(data_rows)

# Save CSV
output_path = 'clean_disease_dataset.csv'
df.to_csv(output_path, index=False)

print(f"✅ Dataset generated: {output_path} | Shape: {df.shape}")
print(f"✅ Symptoms used: {len(all_symptoms)} symptoms, {len(disease_profiles)} diseases.")
