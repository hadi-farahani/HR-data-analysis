# 📊 HR Attrition Analytics Dashboard

A machine learning–powered HR analytics dashboard for predicting employee attrition risk and supporting data-driven HR decision making.

This project provides an interactive dashboard that analyzes employee data, predicts attrition probability, identifies high-risk employees, and visualizes key HR performance indicators.

---

## 🚀 Features

✅ Employee attrition prediction using Machine Learning
✅ Risk probability estimation for each employee
✅ High-risk employee identification
✅ Department-level attrition analysis
✅ Interactive dashboard with filtering and risk threshold control
✅ Financial risk estimation for potential attrition
✅ Feature importance analysis for managerial insights
✅ Individual employee attrition prediction tool

---

## 🧠 Machine Learning

The project uses:

* Random Forest Classifier
* SMOTE for class imbalance handling
* StandardScaler for feature scaling
* One-hot encoding for categorical variables

---

## 📂 Project Structure

```
HR-data-analysis/
│
├── dashboard.py
├── HR_Attrition_Enhanced.ipynb
├── requirements.txt
├── README.md
└── data/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv
```

---

## ⚙️ Installation

Clone repository or download ZIP:

```
pip install -r requirements.txt
```

---

## ▶️ Run Dashboard

```
streamlit run dashboard.py
```

Then open:

```
http://localhost:8501
```

---

## 📊 Dashboard Capabilities

* KPI visualization (attrition rate, high-risk employees, financial risk)
* Attrition analysis by department
* Risk probability distribution
* High-risk employee table
* Feature importance visualization
* Individual employee attrition prediction

---

## 🎯 Business Value

This dashboard helps organizations:

* Reduce employee turnover
* Identify retention risk early
* Improve HR strategy
* Support data-driven decision making
* Estimate financial impact of attrition

---

## 📦 Dataset

IBM HR Employee Attrition Dataset

---

## 👨‍💻 Author

Mohammad Hadi Farahani

---

## ⭐ Future Improvements

* Model performance optimization
* Explainable AI (SHAP)
* Advanced HR segmentation
* Cloud deployment
* Real-time HR analytics

---
