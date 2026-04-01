def generate_farmer_explanation(disease: str, confidence: float, stage: str) -> dict:
    # 1. Convert confidence
    confidence_pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
    confidence_label = f"{confidence_pct}% sure"

    # 2. Map stage to urgency urgency
    stage_lower = stage.lower() if stage else "moderate"
    if "low" in stage_lower or "early" in stage_lower:
        urgency = "Safe"
    elif "severe" in stage_lower or "high" in stage_lower:
        urgency = "Act Now"
    else:
        urgency = "Needs Attention"

    # 3. Disease-specific knowledge base
    knowledge_base = {
        "rice false smut": {
            "what_is_this": "A fungal disease that transforms individual rice grains into velvety, greenish-black spore balls. It thrives in high humidity and excess nitrogen.",
            "actions": [
                "Remove and safely burn the infected panicles (smut balls) to prevent spreading.",
                "Spray Copper Oxychloride 50WP (2.5g/L) or Propiconazole 25EC (1ml/L) during the booting stage.",
                "Reduce the usage of heavy Nitrogen fertilizers immediately."
            ],
            "impact": "Can cause 5-20% yield loss and significantly reduces the grain quality and market price.",
            "cost": "₹500–₹1500 per acre",
            "safety": ["Wear a mask to avoid inhaling fungal spores", "Wear gloves while handling infected plants", "Avoid spraying during windy hours"]
        },
        "default": {
            "what_is_this": f"Our AI detected signs of {disease}. This condition requires careful local management.",
            "actions": [
                "Isolate or remove affected plant parts.",
                "Apply a recommended broad-spectrum treatment for your specific crop.",
                "Consult local ICAR or agricultural extension officers."
            ],
            "impact": "Potential yield reduction if not treated promptly.",
            "cost": "₹300–₹800 per acre",
            "safety": ["Wear basic PPE (mask and gloves)", "Follow all chemical label instructions carefully"]
        }
    }

    # Match exact or partial names
    db_entry = knowledge_base["default"]
    for key, data in knowledge_base.items():
        if key in disease.lower():
            db_entry = data
            break

    return {
        "title": disease.title() if disease else "Unknown Issue",
        "confidence_label": confidence_label,
        "urgency": urgency,
        "what_is_this": db_entry["what_is_this"],
        "actions": db_entry["actions"],
        "impact": db_entry["impact"],
        "cost": db_entry["cost"],
        "safety": db_entry["safety"]
    }
