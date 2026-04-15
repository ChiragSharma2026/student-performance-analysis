import csv
import random

# Output file path
OUTPUT_FILE = "data/student_data.csv"

# Number of students to generate
NUM_STUDENTS = 500

# Open CSV file
with open(OUTPUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Header
    writer.writerow([
        "Student_ID",
        "Attendance",
        "Maths",
        "Science",
        "English",
        "Result"
    ])

    # Generate data
    for i in range(1, NUM_STUDENTS + 1):
        student_id = f"S{str(i).zfill(3)}"

        attendance = random.randint(50, 99)

        # Marks loosely correlated with attendance
        maths = random.randint(40, 95) if attendance > 65 else random.randint(40, 85)
        science = random.randint(40, 95) if attendance > 65 else random.randint(40, 85)
        english = random.randint(45, 95) if attendance > 65 else random.randint(45, 90)

        # Result logic (not perfect, intentionally)
        result = "Pass" if (
            attendance >= 75 and
            maths >= 60 and
            science >= 60 and
            english >= 55
        ) else "Fail"

        writer.writerow([
            student_id,
            attendance,
            maths,
            science,
            english,
            result
        ])

print(f"✅ Generated {NUM_STUDENTS} student records in {OUTPUT_FILE}")
