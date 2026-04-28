# (Updated crop.py - fixed predict and helper issues)

# --- ONLY RELEVANT FIXED PARTS SHOWN BELOW ---

def safe_generate_summary(crop, prob, req):
    try:
        exp = generate_ai_explanation(
            crop,
            float(prob),
            "moderate",
            "en"
        )
        return exp.get("summary", "No explanation available")
    except Exception as e:
        print("SUMMARY ERROR:", e)
        return "Explanation unavailable"


@app.post("/predict")
def predict(req: PredictRequest, current_user: str = Depends(get_current_user)):
    warnings = []
    if req.temperature > 60:
        warnings.append("Extreme heat detected.")
    if req.ph < 0 or req.ph > 14:
        warnings.append("pH must be 0-14.")

    row = {
        "N": req.nitrogen, "P": req.phosphorus, "K": req.potassium,
        "temperature": req.temperature, "humidity": req.humidity,
        "ph": req.ph, "rainfall": req.rainfall
    }

    X = pd.DataFrame([row])
    X = X[feature_names]

    try:
        probs = model.predict_proba(X)[0]
        classes = model.classes_
    except Exception:
        pred = model.predict(X)
        classes = pred
        probs = [1.0]

    idx = np.argsort(probs)[::-1][:req.top_n]

    explanation = []
    try:
        try:
            importances = model.feature_importances_
        except AttributeError:
            importances = [0.1] * len(feature_names)

        for i, fname in enumerate(feature_names):
            explanation.append({
                "feature": fname,
                "contribution": float(importances[i])
            })

        explanation = sorted(explanation, key=lambda x: x["contribution"], reverse=True)

    except Exception:
        explanation = [{"field": "System", "contribution": 0.1}]

    return {
        "status": "success",
        "metadata": {
            "accuracy": model_accuracy,
            "soil_type": req.soil_type,
            "warnings": warnings or None,
            "explainability": {
                "top_crop_explained": classes[idx[0]],
                "feature_contributions": explanation
            }
        },
        "suggestions": [
            {
                "crop": classes[i],
                "probability": float(probs[i]),
                "economics": CROP_ECONOMICS.get(classes[i].lower(), CROP_ECONOMICS["default"]),
                "summary": safe_generate_summary(classes[i], probs[i], req)
            } for i in idx
        ]
    }
