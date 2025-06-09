import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

# Step 1: Load dataset
data = pd.read_csv('C:/Users/indep/Documents/workspace/Python/disease-prediction-ml/data/clean_disease_dataset.csv')

# Debug: print columns to verify target column name
print("Columns in dataset:", data.columns)

# Encode target
target_col = 'Disease'
encoder = LabelEncoder()
data[target_col] = encoder.fit_transform(data[target_col])

# Features & Target
X = data.drop(target_col, axis=1)
y = data[target_col]

# Visualize original class distribution
plt.figure(figsize=(18, 8))
sns.countplot(x=y)
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.show()

# Step 2: Split data BEFORE resampling (to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")

# Step 3: Apply RandomOverSampler ONLY on training data
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Visualize resampled class distribution
plt.figure(figsize=(18, 8))
sns.countplot(x=y_train_resampled)
plt.title("Disease Class Distribution After Resampling (Training Set)")
plt.xticks(rotation=90)
plt.show()

# Step 4: Cross-validation on resampled training set
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

cv_scoring = 'accuracy'
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=stratified_kfold, scoring=cv_scoring, n_jobs=-1)
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"CV Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f}")

# Step 5: Train individual models on resampled training set
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

nb_model = GaussianNB()
nb_model.fit(X_train_resampled, y_train_resampled)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train_resampled, y_train_resampled)

# Step 6: Evaluate models on TEST set
def plot_conf_matrix(y_true, preds, title):
    cf_matrix = confusion_matrix(y_true, preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# SVM
svm_preds = svm_model.predict(X_test)
plot_conf_matrix(y_test, svm_preds, "Confusion Matrix - SVM (Test Set)")
print(f"SVM Test Accuracy: {accuracy_score(y_test, svm_preds) * 100:.2f}%")

# Naive Bayes
nb_preds = nb_model.predict(X_test)
plot_conf_matrix(y_test, nb_preds, "Confusion Matrix - Naive Bayes (Test Set)")
print(f"Naive Bayes Test Accuracy: {accuracy_score(y_test, nb_preds) * 100:.2f}%")

# Random Forest
rf_preds = rf_model.predict(X_test)
plot_conf_matrix(y_test, rf_preds, "Confusion Matrix - Random Forest (Test Set)")
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, rf_preds) * 100:.2f}%")

# XGBoost
xgb_preds = xgb_model.predict(X_test)
plot_conf_matrix(y_test, xgb_preds, "Confusion Matrix - XGBoost (Test Set)")
print(f"XGBoost Test Accuracy: {accuracy_score(y_test, xgb_preds) * 100:.2f}%")

# LightGBM
lgbm_preds = lgbm_model.predict(X_test)
plot_conf_matrix(y_test, lgbm_preds, "Confusion Matrix - LightGBM (Test Set)")
print(f"LightGBM Test Accuracy: {accuracy_score(y_test, lgbm_preds) * 100:.2f}%")

# Combined model (majority vote)
final_preds = []
for i, j, k, l, m_ in zip(rf_preds, xgb_preds, lgbm_preds, svm_preds, nb_preds):
    vote = mode([i, j, k, l, m_], keepdims=False)
    final_preds.append(vote.mode.item() if np.ndim(vote.mode) > 0 else vote.mode)
final_preds = np.array(final_preds)

plot_conf_matrix(y_test, final_preds, "Confusion Matrix - Combined Model (Test Set)")
print(f"Combined Model Test Accuracy: {accuracy_score(y_test, final_preds) * 100:.2f}%")

# Step 7: Save models & encoder
model_dir = 'models'

try:
    os.makedirs(model_dir, exist_ok=True)
    print(f"Directory '{model_dir}' is ready.")
except Exception as e:
    print(f"Failed to create directory '{model_dir}': {e}")

try:
    joblib.dump(rf_model, os.path.join(model_dir, 'rf_model.pkl'))
    print("Random Forest model saved.")
except Exception as e:
    print(f"Error saving Random Forest model: {e}")

try:
    joblib.dump(nb_model, os.path.join(model_dir, 'nb_model.pkl'))
    print("Naive Bayes model saved.")
except Exception as e:
    print(f"Error saving Naive Bayes model: {e}")

try:
    joblib.dump(svm_model, os.path.join(model_dir, 'svm_model.pkl'))
    print("SVM model saved.")
except Exception as e:
    print(f"Error saving SVM model: {e}")

try:
    joblib.dump(xgb_model, os.path.join(model_dir, 'xgb_model.pkl'))
    print("XGBoost model saved.")
except Exception as e:
    print(f"Error saving XGBoost model: {e}")

try:
    joblib.dump(lgbm_model, os.path.join(model_dir, 'lgbm_model.pkl'))
    print("LightGBM model saved.")
except Exception as e:
    print(f"Error saving LightGBM model: {e}")

try:
    joblib.dump(encoder, os.path.join(model_dir, 'label_encoder.pkl'))
    print("Label encoder saved.")
except Exception as e:
    print(f"Error saving label encoder: {e}")

try:
    joblib.dump(list(X.columns.values), os.path.join(model_dir, 'symptoms_list.pkl'))
    print("Symptoms list saved.")
except Exception as e:
    print(f"Error saving symptoms list: {e}")

print("Models saved successfully!")
