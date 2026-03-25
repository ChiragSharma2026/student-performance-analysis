# Student Performance Analysis and Prediction System

## Overview
This project analyzes student academic performance using a simulated dataset and applies a machine learning model to predict whether a student will pass or fail. The goal is to demonstrate the complete data science workflow, including data generation, preprocessing, analysis, visualization, and model evaluation.

---

## Dataset
- The dataset is **synthetically generated** to simulate realistic academic patterns.
- Real student data was not used due to privacy concerns.
- Features include:
  - Attendance percentage
  - Marks in Maths, Science, and English
  - Final result (Pass / Fail)

A separate script is used to generate the dataset programmatically.

---

## Technologies Used
- Python  
- pandas, NumPy  
- matplotlib  
- scikit-learn (Logistic Regression)  

---

## Workflow
1. **Data Generation**
   - `generate_data.py` creates a realistic student dataset with multiple academic features.

2. **Data Analysis**
   - Summary statistics and exploratory data analysis are performed using pandas.

3. **Visualization**
   - Relationships between attendance and subject-wise performance are visualized using scatter plots.

4. **Machine Learning**
   - Logistic Regression is used to predict Pass/Fail outcomes.
   - Class imbalance is handled using `class_weight="balanced"`.

5. **Evaluation**
   - Model performance is evaluated using accuracy and classification metrics.

---

## Project Structure
student-performance-analysis/
│
├── data/
│   └── student_data.csv
│
├── generate_data.py
├── analysis.py
└── README.md


---

## How to Run the Project

### Step 1: Generate the Dataset
```bash
python generate_data.py

### Step 2: Run Analysis and Prediction
python analysis.py


