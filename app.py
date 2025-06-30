
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the second best model and label encoder
model = joblib.load("soft_voting_clf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        # Collect form data
        email_text = request.form["email_text"]
        email_version = request.form["email_version"]
        hour = int(request.form["hour"])
        weekday = request.form["weekday"]
        user_country = request.form["user_country"]
        user_past_purchases = int(request.form["user_past_purchases"])

        # Prepare input as DataFrame for pipeline
        input_df = pd.DataFrame({
            'email_text': [email_text],
            'email_version': [email_version],
            'hour': [hour],
            'weekday': [weekday],
            'user_country': [user_country],
            'user_past_purchases': [user_past_purchases]
        })

        # Predict using the loaded model (pipeline)
        pred_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]

        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
