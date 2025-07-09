# 🧠 Project Approach: Label Order Sensitivity in Multiclass Classification

This document outlines the thought process, experimental design, and insights gained during the course of this project, which investigates the impact of **label ordering** on **multiclass classification performance**, particularly in **tree-based models** like LightGBM, XGBoost, and Random Forest.

---

## 1. 🎯 Problem Definition

**Objective:** Predict obesity levels (7 classes) based on health and lifestyle features from the UCI Obesity dataset. Beyond basic classification, explore whether the **numeric encoding of categorical target labels** affects model performance.

**Core Hypothesis:** Tree-based models are not invariant to label ordering, and different class label mappings can alter learning behavior — particularly for imbalanced and ordinal classes.

---

## 2. 🧹 Data Preprocessing

- **Dataset**: UCI Obesity dataset with 2111 samples
- **Target variable**: `NObeyesdad` (7 categories)

### Steps:

- Dropped irrelevant columns (e.g., `id`)
- Engineered BMI feature from weight and height
- Binary encoding for yes/no features (e.g., `FAVC`, `SMOKE`)
- Ordinal encoding for features like `CAEC`, `CALC`, `MTRANS`
- Applied `StandardScaler` to numerical features




## 3. 🧪 Label Mapping Experiments

### Baseline Mapping:

```python
label_map = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
```

### Variants Tested:

- Randomized mappings 
- Completely shuffled mappings

### Observation:

Changing the label order, while keeping everything else constant, **impacted minority class recall and overall macro F1-score**, especially in tree-based models.

---

## 4. ⚙️ Models Trained

- `RandomForestClassifier`
- `LGBMClassifier`
- `XGBClassifier`
- **Soft Voting Ensemble** (both unweighted and weighted)

Each model was trained and evaluated under multiple label mapping scenarios.

---

## 5. 🎯 Evaluation Metrics

- **Per-class F1-score**
- **Macro F1-score**
- **Confusion Matrix** (with labels kept in original medical order for readability)


---

## 6. 🔍 Key Findings

- **Label order impacts tree-based classifiers**, even though it's not supposed to.
- **Soft voting** ensembles can help, but are also sensitive to individual model biases.
- **Minority class recall** was significantly boosted in some randomized mappings — an accidental discovery that became a core insight.

## 7. 🧪 Hyperparameter Tuning

### Performed GridSearchCV on:

- `RandomForestClassifier`: `n_estimators`, `max_depth`
- `LGBMClassifier`: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`
- `XGBClassifier`: similar to LGBM 

## 8. 🧠 Reflections

- This was my first ML project and what began as routine classification evolved into a semi-research problem.
- I realized the danger of **metric obsession** and learned to focus on meaningful insights over tiny performance gains.
- The biggest win was turning an accidental label mapping bug into a hypothesis and then validating it experimentally.

---

## ✅ Conclusion

Label ordering — an often-overlooked aspect in multiclass classification — has a non-trivial impact on model learning, especially in tree ensembles. This project revealed that small implementation details can drastically change outcomes and need careful treatment in sensitive domains like healthcare.

---



