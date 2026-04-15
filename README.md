# Student Performance Analysis & Risk Detection System

## Overview
An end-to-end data science system that analyzes student academic performance using real-world data from the **UCI Machine Learning Repository** (Cortez et al., 2008) to identify at-risk students and recommend early interventions.

This is not just a prediction script — it's a decision support system built for school administration.

---

## Dataset
- **Source:** UCI ML Repository — [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Citation:** Cortez, P. (2008). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
- **Original size:** 395 Portuguese secondary school students
- **Features used:** Attendance (derived from absences), Maths (G3), Science (G2), English (G1), Result (Pass if G3 ≥ 10)
- **Realistic pass rate:** ~67%

> Raw data mapped via `converter.py` from `student-mat.csv` to pipeline-compatible format.

---

## Key Findings (from UCI data)
- Students with <60% attendance have a **52.1% failure rate**
- **Average Score** is the stronger predictor of passing (r=0.76) — not attendance
- Average marks gap between Fail and Pass students: **30 points**
- High attendance band still had 94 failures — proves attendance alone is insufficient

---

## Technologies
- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, cross-validation)
- matplotlib, seaborn
- Streamlit (web dashboard)

---
## Project Structure
```
student-performance-analysis/
├── data/
│   └── student_data.csv
├── analysis.py
├── app.py
├── converter.py
├── generate_data.py
├── student-mat.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Features
- **Feature Engineering** — Total, Average, WeightedScore (Maths×0.4), AttXAvg interaction, SubjectStd
- **Model Comparison** — Logistic Regression vs Random Forest with 5-fold stratified CV
- **Threshold Tuning** — Optimized for at-risk recall, not raw accuracy
- **Error Analysis** — False negatives examined with avg attendance and score
- **Streamlit Dashboard** — Upload CSV, view insights, compare models, predict students, download HTML report

---

## How to Run

### Option 1: Streamlit App (recommended)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: CLI Pipeline
```bash
python analysis.py
```

### Re-convert raw UCI data (optional)
```bash
python converter.py
```

---

## Deployment
Live on Streamlit Cloud: `(https://student-risk-analyzer.streamlit.app/)`
