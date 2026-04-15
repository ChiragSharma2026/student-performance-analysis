# 🎓 Student Performance Analysis & Risk Detection System

## Overview

An end-to-end data science project that analyzes student academic performance using real-world data from the **UCI Machine Learning Repository** to identify at-risk students and enable early intervention.

This project goes beyond basic prediction — it acts as a **prototype decision support tool** for school administration.

---

## Problem Statement

Schools often identify struggling students too late.
This project aims to proactively detect at-risk students using data-driven insights, allowing timely academic or attendance-based interventions.

---

## Dataset

* **Source:** UCI ML Repository — Student Performance Dataset
* **Citation:** Cortez, P. (2008). Student Performance
* **Size:** 395 Portuguese secondary school students
* **Features used:**

  * Attendance (derived from absences)
  * Maths (G3), Science (G2), English (G1)
  * Result (Pass if G3 ≥ 10)
* **Pass rate:** ~67%

> Raw data is converted using `converter.py` into a pipeline-ready dataset.

---

## Key Findings

* Students with <60% attendance have a **52.1% failure rate**
* **Average Score** is a stronger predictor of passing (r ≈ 0.76) than attendance
* Average marks gap between Fail and Pass students: **~30 points**
* High attendance alone does not guarantee success → academic performance matters more

---

## Technologies Used

* Python, pandas, NumPy
* scikit-learn (Logistic Regression, Random Forest, cross-validation)
* matplotlib, seaborn
* Streamlit (interactive dashboard)

---

## Project Structure

```
student-performance-analysis/
├── data/
│   └── student_data.csv
├── app.py
├── analysis.py
├── converter.py
├── student-mat.csv
├── requirements.txt
└── README.md
```

---

## Features

* **Feature Engineering**

  * Total, Average, WeightedScore, AttXAvg interaction, SubjectStd

* **Model Comparison**

  * Logistic Regression vs Random Forest with cross-validation

* **Threshold Tuning**

  * Optimized for detecting at-risk students (recall-focused)

* **Error Analysis**

  * Analysis of misclassified students to understand edge cases

* **Streamlit Dashboard**

  * Upload CSV
  * Visualize insights
  * Compare models
  * Predict student risk
  * Generate downloadable HTML report

* **Automated Report Generation**

  * Produces a professional HTML report with insights, model performance, and recommendations

---

## Model Performance

| Model               | Accuracy | CV F1 Score |
| ------------------- | -------- | ----------- |
| Logistic Regression | ~1.00    | ~0.99       |
| Random Forest       | ~1.00    | ~1.00       |

> ⚠️ Note: High accuracy is influenced by strong correlation between features and target.

---

## How to Run

### Option 1: Streamlit App (Recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: CLI Pipeline

```bash
python analysis.py
```

### Convert Raw Dataset (Optional)

```bash
python converter.py
```

---

## Deployment

Live Demo: *(Add your Streamlit link here after deployment)*

---

## Future Improvements

* Remove data leakage for more realistic model evaluation
* Add real-time student monitoring system
* Integrate database + authentication
* Deploy with API backend

---

## Author

Chirag Sharma
BTech IT | Aspiring Data Analyst / Data Scientist
