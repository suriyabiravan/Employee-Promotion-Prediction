#Import and Warning control
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_preprocessing import preprocess_data
from train_models import get_models
from evaluate import evaluate_model
from predict import predict_employee
#Load
path = "data/train.csv"
X_train, X_test, y_train, y_test, scaler = preprocess_data(path)
#Load Models
models = get_models()
accuracy_scores = []
model_names = []
best_model = None
best_acc = 0
# Train models
for name, model in models.items():
    print("Training model:", name)
    acc = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    accuracy_scores.append(acc)
    model_names.append(name)
    # Track best model
    if acc > best_acc:
        best_acc = acc
        best_model = model
#Bar chart
#Convert to percentage
accuracy_percent = [acc * 100 for acc in accuracy_scores]
#Create DataFrame
results_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracy_percent
})
# Sort models
results_df = results_df.sort_values(by="Accuracy", ascending=False)
#Plot
plt.figure(figsize=(10,6))
sns.barplot(
    x="Accuracy",
    y="Model",
    data=results_df
)
# Add value labels
for index, value in enumerate(results_df["Accuracy"]):
    plt.text(value + 0.5, index, f"{value:.2f}%", va='center')
# Labels & title
plt.title("Model Accuracy Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Accuracy (%)", fontsize=12)
plt.ylabel("Machine Learning Models", fontsize=12)
# Grid
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Best Model:", type(best_model).__name__)
print(f"Best Accuracy: {best_acc * 100:.2f}%")
#Prediction
predict_employee(best_model, scaler)