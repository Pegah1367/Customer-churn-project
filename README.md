https://customer-churn-project-2kyyntbi64gtjmcq9bwdg2.streamlit.app/



you can use it on both mobile and laptop


dataset:
https://huggingface.co/datasets/aai510-group1/telco-customer-churn

**Project Goal:** Build a retention targeting framework that optimizes churn intervention thresholds based on financial ROI and customer lifetime value (LTV) impact, rather than predictive accuracy alone. On the validation set, the baseline churn rate is 26.54%, and the selected threshold achieves a high-risk recall of 86.10%. Assuming a 20% retention success rate, the expected churn after intervention decreases to 21.75%, representing a 4.80% absolute reduction. At a threshold of 0.74, the retention program delivers a 98.14% ROI, with $12,760 in revenue saved against $6,440 in incentive costs. From an LTV perspective, $769,589 of total LTV is at risk, of which an expected $131,027 is preserved. These results are based on business assumptions of a $20 incentive per targeted customer and $200 in revenue saved per successfully retained customer.


# 📊 Customer Churn Prediction – Full-Stack Machine Learning Project

**Profit-Driven Churn Risk Modeling with XGBoost, Business Simulation, and Interactive Application**

---

## 🚀 Project Summary
This project delivers a **full-stack machine learning solution** for **telecom customer churn prediction**, designed not only to predict churn but to **maximize business profit** through risk-based decision making.

Instead of a simple churn / no-churn classifier, customers are segmented into **Low, Medium, and High churn-risk buckets**, enabling targeted retention strategies aligned with real operational workflows.  
The system combines **machine learning, business logic, profit simulation, and an interactive application** into a deployable end-to-end solution.

---

## 🎯 Business Problem
Telecommunication companies face persistent churn rates (20–30% annually), resulting in:
- Lost recurring revenue
- High customer acquisition costs
- Inefficient, broad retention campaigns

**Key challenge:**  
Traditional churn models optimize accuracy, but **accuracy alone does not maximize profit**.

---

## 💡 Solution Overview
This project introduces a **profit-optimized churn prediction system** that:

- Predicts churn probability for each customer
- Maps probabilities into **Low / Medium / High risk buckets**
- Converts predictions into **actionable retention decisions**
- Optimizes decision thresholds based on **financial impact**, not accuracy
- Demonstrates results through an **interactive application**

---

## 🧠 Machine Learning Task
- **Type:** Multi-class classification (Low / Medium / High risk)
- **Entity:** Individual telecom customer
- **Target:** Churn likelihood at next billing cycle
- **Model:** XGBoost (tabular data-optimized)

---

## 🏗️ Architecture (Full-Stack)
Data Source (HuggingFace Telco Dataset)
↓
Data Cleaning & Feature Engineering
↓
XGBoost Model Training
↓
Probability Scoring
↓
Risk Segmentation (Low / Medium / High)
↓
Profit-Based Threshold Optimization
↓
Interactive Application (Streamlit)



---

## 📂 Data
- **Primary Dataset:** Telco Customer Churn (HuggingFace)
- **Features Include:**
  - Demographics (age, senior status, dependents)
  - Service usage (internet, phone, streaming)
  - Contract & billing (tenure, charges, payment type)
  - Engineered ratios and tenure buckets

Data is refreshed **per billing cycle**, enabling batch scoring in production.

---

## 🔧 Feature Engineering
- Tenure buckets
- Charge ratios
- Missing-value imputation (median / “Unknown”)
- One-hot encoding for categorical variables
- Consistent preprocessing pipeline for training and inference

---

## 📈 Model Evaluation (Beyond Accuracy)
Instead of relying only on accuracy or F1-score, this project evaluates **economic impact** using a profit framework:

| Outcome | Business Meaning | Financial Impact |
|------|----------------|----------------|
| TP | High-risk customer retained | +$200 |
| FP | Unnecessary incentive | −$20 |
| FN | Missed churner | −$200 |
| TN | Correctly ignored | $0 |

---

## 💰 Profit-Based Optimization
- Decision threshold selected based on **maximum expected profit**
- Optimal threshold ≈ **0.74**
- Results:
  - **Net profit:** $60,780
  - **Retention ROI:** 98%
  - **Churn rate reduction:** 4.8%
  - **High-risk detection rate:** 86.9%

This demonstrates how **business-aligned thresholds outperform accuracy-based decisions**.

---

## 📊 Key Business KPIs
- Churn Rate Reduction
- High-Risk Detection Accuracy
- Retention ROI
- Cost per Retained Customer
- Lifetime Value (LTV) Preserved

---

## 🖥️ Interactive Application
An interactive **Streamlit dashboard** allows internal users to:
- Input customer attributes
- View churn probability
- See assigned risk bucket
- Receive recommended retention action
- Understand financial implications

Designed for **retention teams, marketing, and customer service agents**.

---

## 📌 Why This Project Matters
This project demonstrates:
- Real-world ML decision making
- Business-driven model evaluation
- Profit optimization instead of metric chasing
- End-to-end deployment thinking
- Clear alignment between **ML, finance, and operations**

It reflects **industry-grade applied AI**, not just model training.

---

## 🧩 Skills Demonstrated
- Machine Learning (XGBoost, Imbalanced Data)
- Feature Engineering & Pipelines
- Profit-Based Model Evaluation
- Business KPI Design
- Decision Threshold Optimization
- Explainable ML (SHAP ready)
- Streamlit Application Development
- End-to-End ML System Design

---

## 🔮 Future Improvements
- Add behavioral & usage-level features
- Introduce customer-level LTV weighting
- Segment-specific thresholds
- Real-time scoring integration
- Advanced explainability dashboards (SHAP)
- LightGBM / CatBoost benchmarking



Feel free to explore the repository or reach out via GitHub / LinkedIn.

