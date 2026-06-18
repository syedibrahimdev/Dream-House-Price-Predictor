<div align="center">

# 🏡 Dream House Price Predictor

**End-to-end ML pipeline that predicts residential house prices with 93% accuracy — powered by XGBoost, SHAP explainability, and deployed on Streamlit.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[🚀 Live Demo](#) · [📂 Source Code](https://github.com/syedibrahimdev/Dream-House-Price-Predictor) · [📊 Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

</div>

---

## 🧐 The Problem

Real estate pricing is opaque — buyers and sellers rarely know if a price is fair. This project builds a production-ready ML pipeline that takes 79 house features and predicts the sale price with high accuracy, while also explaining *why* the model made its prediction using SHAP values.

---

## 🖼️ Demo

![Streamlit App Interface](images/demo_01.png)
![Streamlit App Interface](images/demo_02.png)

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R² Score (Test Set) | **0.93** |
| RMSE | **~$21,000** (~10% avg error) |
| Validation Strategy | 3-Fold Cross-Validation |
| Final Model | XGBoost (Hyperparameter Tuned) |
| Tuning Method | RandomizedSearchCV |
| Best Params | n_estimators=1500, learning_rate=0.01 |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧹 Data Cleaning | Missing value imputation, outlier removal (>4000 sqft low-price houses) |
| 🔧 Feature Engineering | Created `TotalBath`, `HouseAge`, `TotalPorchSF` to capture hidden patterns |
| ⚙️ Sklearn Pipeline | `ColumnTransformer` handles scaling + encoding automatically — zero data leakage |
| 🤖 XGBoost Model | Tuned with `RandomizedSearchCV` across 1500 estimators |
| 📉 SHAP Explainability | See exactly which features drove each prediction |
| 🌐 Streamlit App | Interactive UI — input house features, get price estimate instantly |

---

## 🏗️ ML Pipeline
📂 Raw Data (1460 samples, 81 features)

│

▼

┌──────────────────────┐

│  01_EDA.ipynb        │  ← Distribution analysis, correlation heatmap,

│                      │    outlier detection, missing value audit

└──────────┬───────────┘

│

▼

┌──────────────────────┐

│  02_Preprocessing    │  ← Imputation, encoding, feature engineering

│  .ipynb              │    (TotalBath, HouseAge, TotalPorchSF)

└──────────┬───────────┘

│

▼

┌──────────────────────┐

│  03_Modeling.ipynb   │  ← Baseline models (Linear, Ridge, Lasso, RF)

│                      │    Sklearn Pipeline + ColumnTransformer

└──────────┬───────────┘

│

▼

┌──────────────────────┐

│  04_Model_Tuning     │  ← RandomizedSearchCV on XGBoost

│  .ipynb              │    R²=0.93, RMSE=~$21K, 3-Fold CV

└──────────┬───────────┘

│

▼

┌──────────────────────┐

│  final_model.joblib  │  ← Saved model artifact

└──────────┬───────────┘

│

▼

┌──────────────────────┐

│  app.py              │  ← Streamlit UI: input features → price prediction

└──────────────────────┘

---

## 📂 Dataset

- **Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Size:** 1,460 training samples, 81 features
- **Top Predictive Features:** `OverallQual`, `GrLivArea`, `YearBuilt`, `TotalBsmtSF`, `Neighborhood`

---

## 📁 Project Structure
Dream-House-Price-Predictor/

│

├── data/

│   ├── raw/                      # Original Kaggle dataset

│   └── processed/                # Cleaned data ready for training

│

├── notebooks/

│   ├── 01_EDA.ipynb              # Exploratory Data Analysis

│   ├── 02_Preprocessing.ipynb    # Cleaning & feature engineering

│   ├── 03_Modeling.ipynb         # Baseline model comparison

│   └── 04_Model_Tuning.ipynb     # XGBoost hyperparameter tuning

│

├── src/

│   ├── preprocessing.py          # Modular feature engineering logic

│   └── train_pipeline.py         # Script to train and save the model

│

├── models/

│   └── final_model.joblib        # Saved XGBoost model artifact

│

├── images/

│   ├── demo_01.png

│   └── demo_02.png

│

├── app.py                        # Streamlit web application

├── requirements.txt              # Python dependencies

└── README.md

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/syedibrahimdev/Dream-House-Price-Predictor.git
cd Dream-House-Price-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Pipeline | Scikit-Learn (Pipeline + ColumnTransformer) |
| Model | XGBoost |
| Explainability | SHAP |
| Deployment | Streamlit |
| Model Saving | Joblib |

---

## 🗺️ Roadmap

- [x] EDA & data cleaning
- [x] Feature engineering pipeline
- [x] XGBoost model with hyperparameter tuning
- [x] Streamlit web app
- [ ] SHAP explainability page in app
- [ ] Deploy to Streamlit Cloud
- [ ] Add confidence intervals to predictions

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 👨‍💻 Author

**Syed Ibrahim Ahmed**
[![GitHub](https://img.shields.io/badge/GitHub-syedibrahimdev-181717?style=flat&logo=github)](https://github.com/syedibrahimdev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/syedibrahimdev)

---

<div align="center">
  <sub>Built to understand how ML models make real-world decisions</sub>
</div>
