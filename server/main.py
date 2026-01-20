import joblib 

def main():
    # load the model 
    loaded_model, loaded_scaler = joblib.load("./server/rain_prediction.pkl")
    print("Model loaded successfully")
    print("-"*25)

    ###
    features = ["Temperature Min (C)", "Temperature Max (C)", "Wind (Km/h)", "Humidity (%)", "Atmospheric Pressure (hPa)"]
    data = []
    print("Enter the data: (Temperature min, Temperature max, wind speed, humidity, atmospheric pressure)")
    for feature in features:
        data.append(float(input(f"{feature}:\n")))
    
    ### send data to the model 
    # Standardize data 
    standardized_data = loaded_scaler.transform([data])

    prediction = loaded_model.predict(standardized_data)
    confidence = loaded_model.predict_proba(standardized_data)[:, 1]

    # output
    print("-"*25)
    print(f"Rain Prediction:\nPrediction: {'Yes' if prediction[0] == 1 else 'No'}\nConfidence: {'{:0.2f}%'.format(confidence[0] * 100)}")

if __name__ == "__main__":
    main()