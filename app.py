import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
def train_models():
    # Load sample data
    df = pd.read_csv("heart_failure_clinical_records_dataset1.csv")

    # Split data into features (X) and target (y)
    X = df[["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]]
    y = df["DEATH_EVENT"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

     # Train LGBoost model
    lgb_model = LGBMClassifier()
    lgb_model.fit(X_train, y_train)



    # Save the trained models
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    with open('lgboost_model.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)



def load_models():
    # Load the trained models
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_model_loaded = pickle.load(f)

    # Load the trained models
    with open('lgboost_model.pkl', 'rb') as f:
        lgb_model_loaded = pickle.load(f)


    return xgb_model_loaded, lgb_model_loaded

# Train the models before loading
train_models()

# Load the models
xgb_model_loaded, lgb_model_loaded = load_models()

# Create a Voting Classifier
voting_clf = VotingClassifier(estimators=[('xgb', xgb_model_loaded), ('lgboost', lgb_model_loaded)], voting='hard')

# Load sample data
df = pd.read_csv("heart_failure_clinical_records_dataset1.csv")

# Split data into features (X) and target (y)
X = df[["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]]
y = df["DEATH_EVENT"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the VotingClassifier
voting_clf.fit(X_train, y_train)

# Define routes for different pages
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/form")
def form():
    return render_template("form.html")

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data
    if request.method == 'POST':
        age = int(request.form['age'])
        anaemia = int(request.form['anaemia'])
        creatinine_phosphokinase = int(request.form['creatinine_phosphokinase'])
        diabetes = int(request.form['diabetes'])
        ejection_fraction = int(request.form['ejection_fraction'])
        high_blood_pressure = int(request.form['high_blood_pressure'])
        platelets = int(request.form['platelets'])
        serum_creatinine = float(request.form['serum_creatinine'])
        serum_sodium = int(request.form['serum_sodium'])
        sex = int(request.form['sex'])
        smoking = int(request.form['smoking'])
        time = int(request.form['time'])
        values = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]])
        prediction = voting_clf.predict(values)
        print(prediction)
        if prediction==0:
            return render_template('result.html',prediction_text=0)
        elif prediction==1:
            return render_template('result.html',prediction_text=1)


        # Render the result template with prediction



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
