import pandas as pd

df = pd.read_csv("student-mat.csv", sep=";")

df_mapped = pd.DataFrame()

# Attendance stays same
df_mapped["Attendance"] = df["absences"].apply(lambda x: max(40, 100 - (x * 3)))

# FIXED: no G3 in features
df_mapped["Maths"] = (df["G1"] * 5).clip(0, 100)
df_mapped["Science"] = (df["G2"] * 5).clip(0, 100)
df_mapped["English"] = (df["G2"] * 5).clip(0, 100)

# Target (allowed to use G3)
df_mapped["Result"] = df["G3"].apply(lambda x: "Pass" if x >= 10 else "Fail")

df_mapped.to_csv("data/student_data.csv", index=False)

print(df_mapped["Result"].value_counts())
print(df_mapped.head())