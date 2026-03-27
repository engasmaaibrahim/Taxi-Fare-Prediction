from django.shortcuts import render
from joblib import load
import numpy as np
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt

# Load Model and Preprocessing Objects
model = load("model.joblib")
scaler = load("scaler.joblib")
feature_names = load("features.joblib")  

def index(request):
    return render(request, "Fare.html")

@csrf_exempt
@api_view(["POST"])
def predict_view(request):
    try:
        data = request.data
        print("Received Data:", data)  

        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])

        # Compute Derived Features
        input_data["distance_traffic"] = input_data["distance"] * input_data["traffic_condition"]
        input_data["rush_hour"] = input_data["hour"].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
        input_data["speed"] = input_data["distance"] / (input_data["hour"] + 1e-3)
        input_data["trip_duration"] = input_data["distance"] / (input_data["speed"] + 1e-3)

        # Remove `fare_amount` only if present
        if "fare_amount" in input_data.columns:
            input_data = input_data.drop(columns=["fare_amount"])

        # Ensure feature order matches model training
        input_data = input_data.loc[:, feature_names]  

        print("Final Input Columns for Model:", input_data.columns.tolist())  

        # Scale Features
        input_scaled = scaler.transform(input_data)

        # Make Prediction
        prediction = model.predict(input_scaled)[0]

        print("Predicted Fare:", prediction)  
        return Response({"predicted_fare": round(float(prediction), 2)})

    except Exception as e:
        print("Error:", e) 
        return Response({"status": "error", "message": str(e)}, status=400)
