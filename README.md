# 🫀 Heart Disease Classification (Interpretable ML)

This project focuses on predicting the presence of heart disease using classical and interpretable machine learning models.

Since the dataset is medical, **model interpretability is prioritized over raw predictive performance**.  
The goal is not only to achieve good metrics, but also to understand **why** a model makes a particular decision.

---

## 🎯 Project Goal

The main goal of this project is to demonstrate how **interpretable machine learning models** can be effectively applied to a medical classification task while maintaining reasonable predictive performance.

The project focuses on:
- building transparent models,
- understanding feature impact on predictions,
- comparing performance across interpretable algorithms.

---

## ❓ Key Questions Addressed

- Can interpretable models achieve competitive performance on a medical dataset?
- Which clinical features contribute most to heart disease prediction?
- What is the trade-off between interpretability and model performance?
- How do different interpretable models compare in terms of stability and metrics?

---

## 📊 Dataset

- Source: Heart Disease dataset (UCI-style medical dataset)
- Task: **Binary classification**
  - `0` — No heart disease
  - `1` — Presence of heart disease
- Features include demographic, clinical and diagnostic measurements.

---

## 🧪 Project Structure
```
heart-disease_classification/
├── notebooks/
│   ├── 01_heart_disease_EDA.ipynb
│   └── 02_pipeline_model.ipynb
│   
├── src/
│   ├── confusion_matrix_visuals.py
│   └── model_comparison_visuals.py
│   
├── README.md
└── requirements.txt
```

### Notebooks
- **01_heart_disease_EDA.ipynb**  
  Exploratory Data Analysis:
  - feature distributions
  - target balance
  - correlations
  - medical insights

- **02_pipeline_model.ipynb**  
  End-to-end modeling pipeline:
  - preprocessing
  - model training
  - hyperparameter tuning
  - evaluation and comparison

---

## 🧠 Model Selection Rationale

In medical applications, transparency and explainability are critical.  
For this reason, only **interpretable or semi-interpretable models** were used.

The following models are included:

- **Logistic Regression**  
  A strong statistical baseline with fully interpretable coefficients.

- **Decision Tree**  
  Rule-based model that provides clear and intuitive decision logic.

- **Explainable Boosting Machine (EBM)**  
  A state-of-the-art interpretable model that combines high performance with explainability.  
  Commonly used in healthcare and finance.

- **Random Forest**  
  Included as a performance benchmark while still allowing feature importance analysis.

More complex black-box models (e.g. neural networks, non-linear SVMs) were intentionally excluded to preserve interpretability.

---

## 📈 Evaluation Metrics

Model performance is evaluated using metrics suitable for medical classification:

- Accuracy
- Precision / Recall
- Confusion Matrix
- Classification report

Visual comparisons and confusion matrix plots are generated using reusable helper scripts.

---

## 🛠️ Tools & Libraries

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- interpret (for Explainable Boosting Machine)

---

## 📌 Status

🚧 Work in progress  
The project may be extended with additional explainability techniques or model diagnostics.
