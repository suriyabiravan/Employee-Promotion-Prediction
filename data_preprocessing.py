import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
def preprocess_data(path):
    #Load dataset
    data = pd.read_csv(path)
    #Drop unnecessary column
    if "employee_id" in data.columns:
        data = data.drop(columns=["employee_id"])
    #Handle missing values
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())
    #Encode categorical variables
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = le.fit_transform(data[col])
    #Separate features and target
    X = data.drop("is_promoted", axis=1)
    y = data["is_promoted"]

    #Train-Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y  
    )
    #Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Apply SMOTE ONLY on training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    #Return processed data
    return X_train, X_test, y_train, y_test, scaler