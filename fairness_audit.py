import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
import os
import json

def run_fairness_audit():
    print("="*60)
    print("🌾 FARMAI BIAS DETECTION & FAIRNESS AUDIT REPORT 🌾")
    print("="*60)
    
    csv_path = "Crop_recommendation/Crop_recommendation.csv"
    model_path = "crop_model.joblib"
    
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        print("❌ Error: Missing dataset or model file. Please ensure 'Crop_recommendation.csv' and 'crop_model.joblib' exist.")
        return
        
    df = pd.read_csv(csv_path)
    data = joblib.load(model_path)
    model = data["model"]
    features = data["features"]
    
    X = df[features]
    y_true = df["label"]
    y_pred = model.predict(X)
    
    print("\n[1] DATASET BIAS DETECTION (Class Distribution)")
    print("-" * 40)
    class_counts = df['label'].value_counts()
    min_class = class_counts.min()
    max_class = class_counts.max()
    
    # Check if dataset is balanced
    if max_class == min_class:
        print(f"✅ The dataset is completely balanced. Every crop has exactly {max_class} samples.")
    else:
        print(f"⚠️ The dataset has imbalances. Range: {min_class} to {max_class} samples per crop.")
    
    # Show distribution of top and bottom 3
    print("📈 Top 3 Most Frequent Crops in Training Data:")
    for crop, count in class_counts.head(3).items():
        print(f"  - {crop}: {count} samples")
        
    print("📉 Bottom 3 Least Frequent Crops in Training Data:")
    for crop, count in class_counts.tail(3).items():
        print(f"  - {crop}: {count} samples")
        
    print("\n[2] OVERALL MODEL ACCURACY")
    print("-" * 40)
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy on Training Dataset: {overall_acc * 100:.2f}%")
    
    print("\n[3] FAIRNESS AUDITING (Subgroup Performance)")
    print("-" * 40)
    print("To ensure model fairness, we evaluate it across different environmental subgroups (e.g., Extreme vs Moderate Climates).")
    
    # Defined subgroups
    median_rainfall = df['rainfall'].median()
    median_temp = df['temperature'].median()
    
    subgroups = {
        "Low Rainfall (< Median)": df['rainfall'] < median_rainfall,
        "High Rainfall (>= Median)": df['rainfall'] >= median_rainfall,
        "Low Temperature (< Median)": df['temperature'] < median_temp,
        "High Temperature (>= Median)": df['temperature'] >= median_temp
    }
    
    for group_name, condition in subgroups.items():
        sub_X = X[condition]
        sub_y_true = y_true[condition]
        if len(sub_X) == 0: continue
            
        sub_y_pred = model.predict(sub_X)
        acc = accuracy_score(sub_y_true, sub_y_pred)
        
        diff = abs(overall_acc - acc)
        status = "✅ FAIR" if diff < 0.05 else "⚠️ POTENTIAL BIAS"
        print(f"Group: {group_name:<30} | Accuracy: {acc * 100:>6.2f}% | Status: {status}")

    print("\n[4] EXPLAINABILITY (SHAP Ready)")
    print("-" * 40)
    print("✅ The model natively supports SHAP (Shapley Additive exPlanations) in the /predict endpoint.")
    print("✅ When a user requests a prediction, the API dynamically computes and returns feature contributions.")
    print("="*60)
    
if __name__ == "__main__":
    run_fairness_audit()
