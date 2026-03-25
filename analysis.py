import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("data/student_data.csv")
print("Dataset Preview:")
print(df.head())

# 2. Basic statistics
print("\nDataset Statistics:")
print(df.describe())

# 3. Convert Result to numeric
df["Result"] = df["Result"].map({"Pass": 1, "Fail": 0})

# 4. Feature selection
X = df[["Attendance", "Maths", "Science", "English"]]
y = df["Result"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train ML model
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

# 7. Model evaluation
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Visualization
plt.scatter(df["Attendance"], df["Maths"], c=df["Result"])
plt.xlabel("Attendance Percentage")
plt.ylabel("Maths Marks")
plt.title("Attendance vs Maths Performance")
plt.show()

# 9. Prediction on new student (CORRECT, no warning)
print("\nEnter new student details:")

attendance = float(input("Attendance percentage: "))
maths = float(input("Maths marks: "))
science = float(input("Science marks: "))
english = float(input("English marks: "))

new_student = pd.DataFrame(
    [[attendance, maths, science, english]],
    columns=["Attendance", "Maths", "Science", "English"]
)

prediction = model.predict(new_student)

print(
    "\nPrediction:",
    "Pass" if prediction[0] == 1 else "Fail"
)

prediction = model.predict(new_student)

print(
    "\nNew Student Prediction:",
    "Pass" if prediction[0] == 1 else "Fail"
)
