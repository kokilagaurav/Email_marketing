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
        hour = request.form["hour"]
        weekday = request.form["weekday"]
        user_country = request.form["user_country"]
        user_past_purchases = request.form["user_past_purchases"]

        # Debug: print received input
        print(f"Received input:")
        print(f"  email_text: {email_text}")
        print(f"  email_version: {email_version}")
        print(f"  hour: {hour}")
        print(f"  weekday: {weekday}")
        print(f"  user_country: {user_country}")
        print(f"  user_past_purchases: {user_past_purchases}")

        # Prepare input as DataFrame for pipeline
        input_df = pd.DataFrame({
            'email_text': [email_text],
            'email_version': [email_version],
            'hour': [int(hour) if hour else 0],
            'weekday': [weekday],
            'user_country': [user_country],
            'user_past_purchases': [int(user_past_purchases) if user_past_purchases else 0]
        })

        # Debug: print prepared DataFrame
        print(f"Prepared DataFrame:")
        print(input_df)

        # Predict using the loaded model (pipeline)
        pred_encoded = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        
        
        # Debug: print the prediction to console
        print(f"Predicted encoded: {pred_encoded}, Decoded: {prediction}")
        
        return render_template(
            "index.html",
            prediction=prediction,
            email_text=email_text,
            email_version=email_version,
            hour=hour,
            weekday=weekday,
            user_country=user_country,
            user_past_purchases=user_past_purchases
        )

    # For GET, set all fields to empty strings
    return render_template(
        "index.html",
        prediction=None,
        email_text="",
        email_version="",
        hour="",
        weekday="",
        user_country="",
        user_past_purchases=""
    )

if __name__ == "__main__":
    app.run(debug=True)
