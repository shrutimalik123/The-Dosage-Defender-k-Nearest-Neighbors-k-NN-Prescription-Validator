# 💊 The Dosage Defender: k-NN Prescription Validator

An interactive, instance-based Supervised Learning simulation designed to teach the **k-Nearest Neighbors (k-NN)** classification algorithm from scratch. You play as a Clinical Pharmacy Data Scientist configuring a real-time health safety mechanism, utilizing multi-dimensional coordinate geometry to intercept hazardous medication overdoses before prescriptions reach the dispensing counter.

## 🎓 Learning Objectives

This project focuses on teaching:
* **k-Nearest Neighbors (k-NN):** A non-parametric classification algorithm that maps novel instances to a category based on a majority vote of its proximal entries.
* **Instance-Based (Lazy) Learning:** Systems that skip upfront model training loops, instead storing reference records raw and computing classification boundaries on-the-fly.
* **The Hyperparameter *k*:** Managing the structural cutoff of neighborhood size to prevent models from overfitting or getting diluted by global majorities.
* **Distance Metric Vulnerabilities:** Demonstrating how variables with massive scales (e.g., mg dosages) can artificially drown out smaller, equally vital coordinates (e.g., body weight) if normalization is absent.

---

## ✨ Features

* **Clinical Pharmacy Scenario:** Translates core spatial analytics into a relatable healthcare compliance and patient safety validation program.
* **Proximity Matrix Breakdown:** Outputs the exact Euclidean distance vectors calculated between the incoming case and every database record.
* **Interactive Polling Mechanics:** Allows players to manually adjust the hyperparameter *k* to see firsthand how neighborhood scaling shifts voting outcomes.
* **Zero Dependency Architecture:** Written strictly in core Python syntax, implementing proximity checks (`sqrt(Σ(Δx)²_i)`) without relying on specialized analytical matrix packages.

---

## 🚀 How to Run the Game

### 1. Prerequisites
You only need **Python 3** installed.

### 2. Setup and Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/dosage-defender-knn.git](https://github.com/YOUR_USERNAME/dosage-defender-knn.git)
    cd dosage-defender-knn
    ```
2.  **Save the Code:** Save the provided script as `dosage_validator.py`.
3.  **Run the Script:**
    ```bash
    python dosage_validator.py
    ```

### 3. Gameplay Instructions
1.  **Read Patient Baselines:** Review historical patient features (Weight vs. Prescription Volume) alongside their verified safety classifications.
2.  **Set Your Neighborhood Size:** Choose a value for *k*. (Pro tip: Always input an odd number to bypass split decision deadlocks!).
3.  **Audit the Queue:** Trace the engine as an exceptionally high dosage for a low-weight patient triggers the network tracking protocols.
4.  **Examine Neighbor Ballots:** Observe which specific historical cases were pulled into the voting quadrant to cross-examine your automated safety verdict.

---

## 🧠 Code Structure Highlights

### Continuous Coordinate Distance Sorting
The framework monitors metric similarities across clinical dimensions by computing the direct spatial hypotenuse between features.

```python
# Euclidean Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
dist = math.sqrt(
    (test_prescription[0] - case["features"][0]) ** 2 + 
    (test_prescription[1] - case["features"][1]) ** 2
)

