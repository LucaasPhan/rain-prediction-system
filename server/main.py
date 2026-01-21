import joblib 

def main():
    # load the model 
    loaded_pipeline = joblib.load("./server/rain_prediction.pkl")
    print("Model loaded successfully")
    print("-"*25)

    ###
    features = ["Temperature Min (C)", "Temperature Max (C)", "Wind (Km/h)", "Humidity (%)", "Atmospheric Pressure (hPa)"]
    data = []
    print("Enter the data: (Temperature min, Temperature max, wind speed, humidity, atmospheric pressure)")
    for feature in features:
        data.append(float(input(f"{feature}:\n")))
    
    ### send data to the model 
    # Predict class probabilities
    proba = loaded_pipeline.predict_proba(data)

    # Predicted class
    y_pred = loaded_pipeline.predict(data)

    # Confidence of predicted class
    confidence = proba.max(axis=1)

    for i in range(10):
        print(f"Prediction: {"Yes" if y_pred[i] == 1 else "No"} - Confidence: {confidence[i]:.2f}")

if __name__ == "__main__":
    main()