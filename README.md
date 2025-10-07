# 🩺 Disease Prediction Using Machine Learning 🚀

This project is a **Disease Prediction System** built using **Machine Learning** and deployed as a simple **Flask web application**.  
The system takes **patient symptoms** as input and predicts the most likely disease.

---

## 🗂 Project Structure

``` 
DISEASE-PREDICTION-ML/
│
├── .venv/ # Virtual environment
├── data/ # Raw and cleaned datasets
│ ├── clean disease dataset.csv
│ └── improved_disease_dataset.csv
├── models/ # Saved ML models and encoder
│ ├── label_encoder.pkl
│ ├── Igbm_model.pkl
│ ├── nb_model.pkl
│ ├── rf_model.pkl
│ ├── svm_model.pkl
│ ├── symptoms_list.pkl
│ └── xgb_model.pkl
├── src/ # ML scripts
│ ├── predict.py # Script to test prediction
│ └── train_model.py # Script to train and save models
├── templates/ # Flask HTML templates
│ ├── index.html
│ └── result.html
├── app.py # Flask app entry point
├── disease_dataset_generator.py # Dataset preprocessing (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## 💻 How It Works

✅ Reads symptom-disease data from CSV  
✅ Uses RandomOverSampler to handle class imbalance  
✅ Trains **five models**:
- SVM (Support Vector Machine)
- Gaussian Naive Bayes
- Random Forest
- LightGBM
- XGBoost

✅ Evaluates with Stratified K-Fold  
✅ Combines model predictions for better accuracy  
✅ Saves models with `joblib`  
✅ Flask Web Interface to predict diseases based on selected symptoms

---

## 🚀 Installation

### 1️⃣ Clone / Download Project

```bash
git clone https://github.com/BALADURGAG24/disease-prediction-ml.git
cd disease-prediction-ml
```

Or download as ZIP and extract.

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Dataset

Put your `improved_disease_dataset.csv` into the `/data` folder.

---

## 🏋️‍♂️ Training Models

Run this command to train models and save them:

```bash
python src/train_model.py
```

It will:
- Train SVM, Naive Bayes, Random Forest, LightGBM, and XGBoost  
- Save models as `.pkl` files in `/models`  
- Save the Label Encoder  
- Save symptom list  

---

## 🌐 Running the Web App

After training, launch the Flask app:

```bash
python app.py
```

Then open this URL in your browser:

```
http://127.0.0.1:5000/
```

You can select symptoms and see the predicted disease!

---

## 🧪 Command-line Prediction (Optional)

You can also test prediction in command line:

```bash
python src/predict.py
```

---

## Example Output (Web)

| Model                  | Prediction      |
|------------------------|-----------------|
| Random Forest          | Heart Attack    |
| Naive Bayes            | UTI             |
| SVM                    | Impetigo        |
| LightGBM               | Malaria         |
| XGBoost                | Dengue          |
| **Final Combined**     | Heart Attack    |

---

## Requirements

- Python 3.8+  
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn
  - scipy
  - joblib
  - flask
  - lightgbm
  - xgboost

---

## File Descriptions

- **app.py**: Flask web application handling routes and prediction logic.

- **train_model.py**: Script for training and saving ML models including SVM, Naive Bayes, Random Forest, LightGBM, and XGBoost.

- **predict.py**: Script for testing model prediction from command line or script.

- **disease_dataset_generator.py**: Dataset cleaning/preprocessing script.

- **requirements.txt**: Python libraries required to run the project.

- **templates/index.html**: Symptom selection form.

- **templates/result.html**: Displays prediction results.

- **models/**: Contains saved models, encoder, and symptom list for inference.

- **data/**: Contains raw and cleaned datasets used for training.

---

## Author & Contribution

- **Author:** BALADURGA G  
- **Email:** baladurga2412@gmail.com  
- **GitHub:** [github.com/BALADURGAG24](https://github.com/BALADURGAG24)

Contributions are welcome!  
Feel free to fork the repository, create branches, and submit pull requests.  
Please follow the code style and include tests if applicable.

---

## Forking this Repository

To contribute or modify this project, fork it on GitHub:

1. Click the **Fork** button on the repository page.  
2. Clone your fork locally:  
   ```bash
   git clone https://github.com/BALADURGAG24/disease-prediction-ml.git
   ```  
3. Make your changes, commit, and push.  
4. Create a Pull Request to merge your changes back.

---

## TODO (Future Improvements)

- Improve hyperparameter tuning  
- Add support for saving user history  
- Deploy on cloud (Heroku / AWS / Azure)  

---

## Credits

- Inspired by various Disease Prediction datasets on Kaggle  
- Built with ❤️ using Python and Flask

---

## License

MIT License — Free to use for educational purposes!
