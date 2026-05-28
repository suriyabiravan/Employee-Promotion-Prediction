import pandas as pd
def predict_employee(model, scaler):
    # Assign Employee ID 
    employee_id = 1001
    # Employee details
    employee_data = {
        "department": 2,
        "region": 10,
        "education": 2,
        "gender": 1,
        "recruitment_channel": 1,
        "no_of_trainings": 5,
        "age": 30,
        "previous_year_rating": 5,
        "length_of_service": 6,
        "awards_won?": 1,
        "avg_training_score": 90
    }
    # Convert to DataFrame
    df = pd.DataFrame([employee_data])
    # Scale
    scaled_data = scaler.transform(df)
    # Prediction
    prediction = model.predict(scaled_data)
    # Probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_data)[0][1]
    else:
        prob = None
    print("\nEmployee Details")
    print(f"Employee ID: {employee_id}")

    for key, value in employee_data.items():
        print(f"{key}: {value}")
    print("\n Prediction Result ")
    if prob is not None:
        print(f"Promotion Probability: {prob * 100:.2f}%")
    if prediction[0] == 1:
        print("Employee WILL be promoted")
    else:
        print("Employee will NOT be promoted")