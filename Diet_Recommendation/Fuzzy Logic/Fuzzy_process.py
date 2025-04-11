import numpy as np
import skfuzzy as fuzz

# Step 1: Fuzzify user input
def fuzzify_user_input(age_val, bmi_val, activity_val, pref_val):
    age = np.arange(15, 81, 1)
    age_young = fuzz.trimf(age, [15, 20, 35])
    age_middle = fuzz.trimf(age, [30, 45, 60])
    age_elderly = fuzz.trimf(age, [55, 70, 80])

    age_membership = {
        "Young": round(fuzz.interp_membership(age, age_young, age_val), 3),
        "Middle-aged": round(fuzz.interp_membership(age, age_middle, age_val), 3),
        "Elderly": round(fuzz.interp_membership(age, age_elderly, age_val), 3)
    }

    bmi = np.arange(10, 41, 0.5)
    bmi_under = fuzz.trimf(bmi, [10, 15, 18.5])
    bmi_normal = fuzz.trimf(bmi, [18.5, 22, 25])
    bmi_over = fuzz.trimf(bmi, [24, 27, 30])
    bmi_obese = fuzz.trimf(bmi, [29, 35, 40])

    bmi_membership = {
        "Underweight": round(fuzz.interp_membership(bmi, bmi_under, bmi_val), 3),
        "Normal": round(fuzz.interp_membership(bmi, bmi_normal, bmi_val), 3),
        "Overweight": round(fuzz.interp_membership(bmi, bmi_over, bmi_val), 3),
        "Obese": round(fuzz.interp_membership(bmi, bmi_obese, bmi_val), 3)
    }

    activity = np.arange(1, 6, 1)
    activity_sedentary = fuzz.trimf(activity, [1, 1, 3])
    activity_moderate = fuzz.trimf(activity, [2, 3, 4])
    activity_active = fuzz.trimf(activity, [3, 5, 5])

    activity_membership = {
        "Sedentary": round(fuzz.interp_membership(activity, activity_sedentary, activity_val), 3),
        "Moderate": round(fuzz.interp_membership(activity, activity_moderate, activity_val), 3),
        "Active": round(fuzz.interp_membership(activity, activity_active, activity_val), 3)
    }

    pref = np.arange(1, 5, 1)
    pref_vegan = fuzz.trimf(pref, [1, 1, 2])
    pref_veg = fuzz.trimf(pref, [1, 2, 3])
    pref_egg = fuzz.trimf(pref, [2, 3, 4])
    pref_nonveg = fuzz.trimf(pref, [3, 4, 4])

    pref_membership = {
        "Vegan": round(fuzz.interp_membership(pref, pref_vegan, pref_val), 3),
        "Vegetarian": round(fuzz.interp_membership(pref, pref_veg, pref_val), 3),
        "Eggitarian": round(fuzz.interp_membership(pref, pref_egg, pref_val), 3),
        "Non-Veg": round(fuzz.interp_membership(pref, pref_nonveg, pref_val), 3)
    }

    return {
        "Age": age_membership,
        "BMI": bmi_membership,
        "Activity Level": activity_membership,
        "Dietary Preference": pref_membership
    }

# Step 2: Define rules and inference
def infer_diet_recommendation(fuzzified_input):
    age = fuzzified_input["Age"]
    bmi = fuzzified_input["BMI"]
    activity = fuzzified_input["Activity Level"]
    pref = fuzzified_input["Dietary Preference"]

    recommendation = np.arange(0, 11, 1)
    rec_low = fuzz.trimf(recommendation, [0, 0, 5])
    rec_moderate = fuzz.trimf(recommendation, [3, 5, 7])
    rec_high = fuzz.trimf(recommendation, [5, 10, 10])

    # --- Existing Rules ---
    rule1 = np.fmin(np.fmin(age["Young"], activity["Active"]), bmi["Underweight"])  # High
    rule2 = np.fmin(np.fmin(age["Middle-aged"], activity["Moderate"]), bmi["Normal"])  # Moderate
    rule3 = np.fmin(np.fmin(age["Elderly"], activity["Sedentary"]), bmi["Obese"])  # Low
    rule4 = np.fmin(bmi["Overweight"], activity["Moderate"])  # Moderate
    rule5 = np.fmin(np.fmin(age["Young"], activity["Sedentary"]), bmi["Obese"])  # Low
    rule6 = np.fmin(activity["Active"], bmi["Normal"])  # High
    rule7 = np.fmin(np.fmin(age["Elderly"], activity["Active"]), bmi["Normal"])  # Moderate
    rule8 = np.fmin(np.fmin(age["Middle-aged"], activity["Sedentary"]), bmi["Overweight"])  # Low
    rule9 = np.fmin(np.fmin(age["Young"], activity["Moderate"]), bmi["Normal"])  # High
    rule10 = np.fmin(np.fmin(age["Middle-aged"], activity["Active"]), bmi["Underweight"])  # Moderate
    rule11 = np.fmin(np.fmin(age["Elderly"], activity["Moderate"]), bmi["Obese"])  # Low
    rule12 = np.fmin(np.fmin(age["Young"], activity["Active"]), bmi["Normal"])  # High

    # --- New Rules including Dietary Preference ---
    rule13 = np.fmin(np.fmin(pref["Vegan"], bmi["Underweight"]), activity["Active"])  # Moderate (protein concern)
    rule14 = np.fmin(np.fmin(pref["Non-Veg"], bmi["Obese"]), activity["Sedentary"])  # Low (high calorie caution)
    rule15 = np.fmin(np.fmin(pref["Vegetarian"], bmi["Normal"]), activity["Moderate"])  # High
    rule16 = np.fmin(np.fmin(pref["Eggitarian"], bmi["Normal"]), activity["Active"])  # High

    # Apply to output fuzzy sets
    out_low = np.fmax(rule3, np.fmax(rule5, np.fmax(rule8, np.fmax(rule11, rule14))))
    out_moderate = np.fmax(rule2, np.fmax(rule4, np.fmax(rule7, np.fmax(rule10, rule13))))
    out_high = np.fmax(rule1, np.fmax(rule6, np.fmax(rule9, np.fmax(rule12, np.fmax(rule15, rule16)))))

    # Aggregate output
    aggregated = np.fmax(
        np.fmax(out_low * rec_low, out_moderate * rec_moderate),
        out_high * rec_high
    )

    result = fuzz.defuzz(recommendation, aggregated, 'centroid')
    return round(result, 2)

# Step 3: Run system
if __name__ == "__main__":
    # Test input
    age_val = 28
    bmi_val = 23
    activity_val = 4
    pref_val = 2  # Vegetarian

    fuzzified = fuzzify_user_input(age_val, bmi_val, activity_val, pref_val)
    score = infer_diet_recommendation(fuzzified)

    print("Fuzzified Input:")
    for key, val in fuzzified.items():
        print(f"{key}: {val}")

    print(f"\nDiet Recommendation Score (0-Low to 10-High): {score}")
