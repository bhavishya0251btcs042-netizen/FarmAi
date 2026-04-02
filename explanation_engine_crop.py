IDEAL_CONDITIONS = {
    "rice": {"N": 80, "P": 48, "K": 40, "temp": 23.7, "rain": 236, "ph": 6.4},
    "maize": {"N": 78, "P": 48, "K": 20, "temp": 22.4, "rain": 85, "ph": 6.2},
    "wheat": {"N": 92, "P": 50, "K": 49, "temp": 19.8, "rain": 76, "ph": 6.8},
    "cotton": {"N": 118, "P": 46, "K": 20, "temp": 24.0, "rain": 80, "ph": 6.9},
    "sugarcane": {"N": 124, "P": 65, "K": 50, "temp": 27.6, "rain": 201, "ph": 6.9},
    "tea": {"N": 100, "P": 51, "K": 51, "temp": 20.5, "rain": 196, "ph": 5.0},
    "coffee": {"N": 101, "P": 29, "K": 30, "temp": 25.5, "rain": 158, "ph": 6.8},
    "potato": {"N": 101, "P": 50, "K": 50, "temp": 17.4, "rain": 88, "ph": 6.0},
    "tomato": {"N": 15, "P": 25, "K": 20, "temp": 22.0, "rain": 100, "ph": 6.5}, # Approximated
    "default": {"N": 50, "P": 50, "K": 50, "temp": 25.0, "rain": 100, "ph": 6.5}
}

def generate_explanation(crop, compatibility, temp, rainfall, ph, n, p, k, yield_range):
    """
    Generates a farmer-friendly explanation for crop recommendation.
    """
    crop_name = crop.title()
    ideal = IDEAL_CONDITIONS.get(crop.lower(), IDEAL_CONDITIONS["default"])
    
    # 1. Suitability & Yield Level
    if compatibility > 0.70:
        base = f"{crop_name} is highly suitable for your land 🌱"
        reason_connector = "because of"
    elif compatibility > 0.40:
        base = f"{crop_name} can grow in your field ☀️, with moderate yield"
        reason_connector = "due to"
    else:
        base = f"{crop_name} is not ideal but possible with extra care 💧"
        reason_connector = "due to"

    # 2. Reason Logic (Find 1-2 key reasons)
    reasons = []
    
    if rainfall < ideal["rain"] * 0.7: reasons.append("low rainfall")
    elif rainfall > ideal["rain"] * 1.3: reasons.append("good water availability")
    
    if ph < 5.5: reasons.append("acidic soil")
    elif ph > 7.5: reasons.append("alkaline soil")
    else: reasons.append("favorable soil pH")
        
    if n < ideal["N"] * 0.6: reasons.append("low nitrogen")
    elif n > ideal["N"] * 0.9: reasons.append("balanced soil nutrients")

    # Pick 2 reasons for brevity
    final_reasons = reasons[:2]
    reason_str = " and ".join(final_reasons)

    # 3. Final Flowing Sentence (Yield Range Mentioned)
    if "moderate" in base or "not ideal" in base:
       # For lower suitability, integrate yield more naturally
       explanation = f"{base} {reason_connector} {reason_str}. Expected yield: {yield_range}."
    else:
       # For high suitability, be more positive
       explanation = f"{base} {reason_connector} {reason_str}. You can expect around {yield_range}."

    return explanation
