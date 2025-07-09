# smart-obesity-predictor
Predicting and understanding obesity levels using explainable ensemble learning

# 🔍 Label Order Sensitivity in Multiclass Tree-Based Classification  
### 📊 An Experimental Analysis on UCI Obesity Dataset Using Random Forest, LightGBM, and XGBoost

---

## 📖 Overview

This project explores a **rarely discussed phenomenon** in multiclass classification:  
👉 *Does the numeric order of class labels affect model performance in tree-based classifiers?*

Using the UCI Obesity dataset, I demonstrate that **the way class labels are encoded** (even when class names remain the same) can **significantly impact recall**, especially for underrepresented classes.

---

## 🧠 Motivation

In typical multiclass classification problems, it’s assumed that class label IDs (e.g., 0–6) are nominal and arbitrary. But during experimentation with obesity-level prediction, I observed:
> 🚨 Shuffling class label mappings changed the F1 scores — *without touching data, features, or model parameters.*

This sparked a deeper exploration of **label order sensitivity**, its impact on tree-based models, and how it interacts with ensemble learning.

---

## 🗂 Dataset: UCI Obesity Dataset

- **Samples**: 2111 rows
- **Target**: 7-class categorical label (`NObeyesdad`)
- **Features**: 17 (including categorical + numerical)
- **Classes**:
  - Insufficient_Weight  
  - Normal_Weight  
  - Overweight_Level_I  
  - Overweight_Level_II  
  - Obesity_Type_I  
  - Obesity_Type_II  
  - Obesity_Type_III



## ⚙️ Approach

### 1. 🧹 Preprocessing
- Converted categorical variables to ordinal numeric representations
- Created new BMI feature from height & weight
- Applied `StandardScaler` on numerical features


### 2. 🔁 Experiment: Label Mapping Variations
Ran identical pipelines with:
- **Original label mapping** (ordinal: underweight → obesity)
- **Randomized mappings** (e.g., swapping class 3 and 6)
  
Tracked model behavior across mappings to detect variance in class performance — particularly minority class **recall and F1-score**.

### 3. 🌲 Models Used
- `RandomForestClassifier`
- `XGBClassifier`
- `LGBMClassifier`
- Ensemble: **Soft Voting Classifier** (unweighted and weighted variants)

### 4. 🔍 Evaluation
- Per-class **F1, Precision, Recall**
- **Macro-averaged F1**
- Confusion Matrices (shown with fixed label order for clarity)
- Grid Search for tuning `n_estimators`, `max_depth`

---

## 📈 Key Insight

> 💡 **Class label order matters in tree-based models.**

Even though the class labels represent the same categories, reordering their numeric mapping:
- Altered **how splits are formed**
- Caused ensemble models to **treat classes differently**
- Boosted minority class F1 scores in certain mappings

This effect is *especially visible in imbalanced datasets with ordinal target variables* — like medical classification.

---

## 📌 Summary of Contributions

- Built a full ML pipeline for multiclass obesity classification
- Discovered and validated the **impact of label order** on performance
- Tracked minority class metrics across label orderings
- Tuned hyperparameters for improved generalization
- Packaged everything into a reproducible notebook

---

## 📁 Project Structure

