import math

def knn_pharmacy_game():
    # 1. Scenario: Clinical Pharmacy Safety Filter
    print("--- 💊 THE DOSAGE DEFENDER: k-NN PRESCRIPTION VALIDATOR 💊 ---")
    print("Mission: Validate incoming prescriptions against a database of verified patient cases.")
    print("Goal: Find the 'k' closest historical examples to vote on whether a dosage is safe.")

    # 2. Historical Reference Database: [Weight (kg), Dosage (mg)] -> Class (0 = Safe, 1 = Toxic)
    # General trend: Higher weight accommodates higher dosage. Outliers violate this curve.
    history = [
        {"features": [50.0, 100.0], "label": 0, "type": "Verified Safe"},
        {"features": [55.0, 120.0], "label": 0, "type": "Verified Safe"},
        {"features": [85.0, 250.0], "label": 0, "type": "Verified Safe"},
        {"features": [45.0, 300.0], "label": 1, "type": "Toxic Overdose"},
        {"features": [60.0, 350.0], "label": 1, "type": "Toxic Overdose"},
    ]
    
    print("\n--- 🖥️ HISTORICAL PRESCRIPTION CLINICAL LOGS ---")
    for idx, case in enumerate(history):
        print(f"Case {idx+1}: Patient Weight = {case['features'][0]} kg | Dosage = {case['features'][1]} mg -> [{case['type']}]")

    # 3. Game Inputs: Selecting the Hyperparameter 'k' (Number of Neighbors)
    print("\n--- STEP 1: CONFIGURE YOUR NEIGHBOR MATRIX ---")
    print("The hyperparameter 'k' determines how many close historical cases vote on the new prescription.")
    print("Tip: Use an odd number (like 3) to prevent ties during votes!")
    try:
        k = int(input("Enter value for k (Recommended: 3): "))
        if k <= 0 or k > len(history):
            raise ValueError
    except ValueError:
        k = 3

    # 4. Incoming Intercepted Prescription
    test_prescription = [52.0, 280.0] # Low weight, exceptionally high dosage
    print(f"\n--- 🚨 DISPENSING COUNTER: NEW ORDER PENDING VALIDATION ---")
    print(f"Incoming Patient -> Weight: {test_prescription[0]} kg | Prescribed Dosage: {test_prescription[1]} mg")

    # 5. The Math: Distance Matrix Calculation
    # We measure the Euclidean distance between our test patient and every historical record
    print(f"\n--- 🔄 COMPUTING EUCLIDEAN PROXIMITY COALITIONS ---")
    
    distances = []
    for idx, case in enumerate(history):
        # Euclidean Distance: sqrt((x2 - x1)^2 + (y2 - y1)^2)
        dist = math.sqrt(
            (test_prescription[0] - case["features"][0]) ** 2 + 
            (test_prescription[1] - case["features"][1]) ** 2
        )
        distances.append({"index": idx, "distance": dist, "label": case["label"]})
        print(f"Distance to Case {idx+1}: {dist:.2f}")

    # 6. Sorting and Neighborhood Isolation
    # Sort cases from smallest distance (nearest) to largest distance
    distances.sort(key=lambda x: x["distance"])
    
    print(f"\n--- 📊 ISOLATING THE {k} NEAREST NEIGHBORS ---")
    safe_votes = 0
    toxic_votes = 0
    
    for i in range(k):
        neighbor = distances[i]
        print(f"Neighbor {i+1} (Case {neighbor['index']+1}): Distance = {neighbor['distance']:.2f} | Label Code = {neighbor['label']}")
        if neighbor["label"] == 0:
            safe_votes += 1
        else:
            toxic_votes += 1

    print(f"\nTally: Safe Votes = {safe_votes} | Toxic Votes = {toxic_votes}")

    # 7. Consensus Classification Output
    if toxic_votes > safe_votes:
        prediction = 1
        verdict = "❌ TOXIC OVERDOSE RISK (PRESCRIPTION BLOCKED)"
    else:
        prediction = 0
        verdict = "✅ DOSAGE WITHIN SAFE BOUNDS (ORDER APPROVED)"

    print(f"Automated Filter Verdict: {verdict}")

    # 8. Ground Truth Evaluation
    # Mathematically, 280mg for a 52kg patient sits directly inside the historical overdose cluster
    actual_truth = 1 
    
    if prediction == actual_truth:
        print("\n🏆 SUCCESS: Your k-NN safety model intercepted a dangerous clinical overdose!")
        print("The system successfully flagged the prescription for pharmacist review.")
    else:
        print("\n💥 COMPLIANCE BREACH: Medical validation error! Your value for k misclassified the patient risk.")

if __name__ == "__main__":
    knn_pharmacy_game()
