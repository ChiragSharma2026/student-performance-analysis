import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────

def load_and_prepare(path):
    df = pd.read_csv(path)
    df["Result"] = df["Result"].map({"Pass": 1, "Fail": 0})

    # Basic features
    df["Total"] = df["Maths"] + df["Science"] + df["English"]
    df["Average"] = df["Total"] / 3

    # Weighted score — Maths weighted higher (typically stronger predictor)
    df["WeightedScore"] = (df["Maths"] * 0.4) + (df["Science"] * 0.3) + (df["English"] * 0.3)

    # Interaction feature — captures combined effect of attendance + performance
    df["AttXAvg"] = df["Attendance"] * df["Average"]

    # Subject consistency — low std = consistent performer
    df["SubjectStd"] = df[["Maths", "Science", "English"]].std(axis=1)

    # Bands
    df["AttendanceBand"] = pd.cut(df["Attendance"], bins=[0, 60, 75, 100], labels=["Low", "Medium", "High"])
    df["PerformanceBand"] = pd.cut(df["Average"], bins=[0, 40, 60, 100], labels=["Weak", "Average", "Strong"])

    return df

def show_insights(df):
    fail = df[df["Result"] == 0]
    pass_ = df[df["Result"] == 1]

    print("\n─── KEY INSIGHTS ───")
    print(f"Total Students     : {len(df)}")
    print(f"Pass Rate          : {df['Result'].mean()*100:.1f}%")
    print(f"Avg Attendance     → Fail: {fail['Attendance'].mean():.1f}%  | Pass: {pass_['Attendance'].mean():.1f}%")
    print(f"Avg Total Marks    → Fail: {fail['Total'].mean():.1f}      | Pass: {pass_['Total'].mean():.1f}")
    print(f"Avg Weighted Score → Fail: {fail['WeightedScore'].mean():.1f}  | Pass: {pass_['WeightedScore'].mean():.1f}")

    low_att = df[df["Attendance"] < 60]
    low_att_fail = len(low_att[low_att["Result"] == 0])
    if len(low_att) > 0:
        rate = low_att_fail / len(low_att) * 100
        print(f"\n📌 Hard Finding 1: Students with <60% attendance → {rate:.1f}% failure rate ({low_att_fail}/{len(low_att)})")

    att_corr = df["Attendance"].corr(df["Result"])
    avg_corr = df["Average"].corr(df["Result"])
    stronger = "Attendance" if abs(att_corr) > abs(avg_corr) else "Average Score"
    print(f"📌 Hard Finding 2: '{stronger}' correlates more strongly with passing (r={max(abs(att_corr), abs(avg_corr)):.2f})")

    print("\n─── ATTENDANCE BAND vs RESULT ───")
    print(pd.crosstab(df["AttendanceBand"], df["Result"].map({1: "Pass", 0: "Fail"}), margins=True))

def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    }
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n─── MODEL COMPARISON ───")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        results[name] = {"model": model, "pred": y_pred, "acc": acc, "cv_f1": cv_scores.mean()}
        print(f"\n{name}")
        print(f"  Test Accuracy  : {acc:.4f}")
        print(f"  CV F1 (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    best_name = max(results, key=lambda k: results[k]["cv_f1"])
    print(f"\n✅ Best Model by CV F1: {best_name}")
    return results, best_name

def tune_threshold(results, best_name, X_test, y_test, feature_cols):
    print("\n─── THRESHOLD TUNING ───")
    model = results[best_name]["model"]
    probs = model.predict_proba(X_test)[:, 1]

    print(f"{'Threshold':<12} {'Recall(Fail)':<15} {'Precision(Fail)':<18} {'F1(Fail)':<12}")
    print("-" * 57)

    best_threshold = 0.5
    best_recall = 0
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5]:
        custom_pred = (probs >= thresh).astype(int)
        report = classification_report(y_test, custom_pred, target_names=["Fail", "Pass"], output_dict=True)
        recall_fail = report["Fail"]["recall"]
        prec_fail = report["Fail"]["precision"]
        f1_fail = report["Fail"]["f1-score"]
        print(f"{thresh:<12} {recall_fail:<15.4f} {prec_fail:<18.4f} {f1_fail:<12.4f}")
        if recall_fail > best_recall:
            best_recall = recall_fail
            best_threshold = thresh

    print(f"\n✅ Optimal Threshold for at-risk detection: {best_threshold}")
    print(f"   Rationale: Maximizes recall on Fail class → fewer at-risk students missed")

    final_pred = (probs >= best_threshold).astype(int)
    results[best_name]["tuned_pred"] = final_pred
    results[best_name]["best_threshold"] = best_threshold
    results[best_name]["probs"] = probs
    return results

def show_feature_importance(results, best_name, feature_names):
    print("\n─── FEATURE IMPORTANCE ───")
    model = results[best_name]["model"]

    if best_name == "Random Forest":
        imp = model.feature_importances_
        label = "Importance Score"
    else:
        imp = model.coef_[0]
        label = "Coefficient"

    imp_df = pd.DataFrame({"Feature": feature_names, label: imp}).sort_values(label, ascending=False)
    print(imp_df.to_string(index=False))

    top = imp_df.iloc[0]
    bottom = imp_df.iloc[-1]
    print(f"\nInterpretation:")
    print(f"  → '{top['Feature']}' is the strongest positive predictor of passing.")
    print(f"  → '{bottom['Feature']}' has the least/most negative influence on outcome.")
    return imp_df, label

def error_analysis(X_test, y_test, results, best_name):
    y_pred = results[best_name].get("tuned_pred", results[best_name]["pred"])
    misclassified_mask = y_test.values != y_pred
    misclassified = X_test[misclassified_mask].copy()
    misclassified["Actual"] = y_test.values[misclassified_mask]
    misclassified["Predicted"] = y_pred[misclassified_mask]

    false_positives = misclassified[misclassified["Predicted"] == 1]
    false_negatives = misclassified[misclassified["Predicted"] == 0]

    print(f"\n─── ERROR ANALYSIS ───")
    print(f"Total Misclassified : {len(misclassified)} / {len(X_test)}")
    print(f"False Positives (predicted Pass, actually Fail) : {len(false_positives)}")
    print(f"False Negatives (predicted Fail, actually Pass) : {len(false_negatives)}")

    if len(false_negatives) > 0:
        print(f"\nAvg Attendance of missed at-risk students : {false_negatives['Attendance'].mean():.1f}%")
        print(f"Avg Score of missed at-risk students      : {false_negatives['Average'].mean():.1f}")
        print("→ These students slipped through despite weak indicators — borderline cases.")
        print("→ School counselors should manually review students in 60–75% attendance range.")

def visualize(df, imp_df, label, results, best_name, y_test):
    probs = results[best_name]["probs"]
    thresh = results[best_name]["best_threshold"]
    tuned_pred = results[best_name]["tuned_pred"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Student Performance Analysis — Decision Dashboard", fontsize=15, fontweight="bold")

    # 1. Attendance vs Result
    sns.boxplot(x="Result", y="Attendance", data=df, ax=axes[0][0], palette=["#e74c3c", "#2ecc71"])
    axes[0][0].set_xticklabels(["Fail", "Pass"])
    axes[0][0].set_title("Attendance Distribution by Result")
    axes[0][0].set_ylabel("Attendance %")

    # 2. Weighted Score vs Result
    sns.boxplot(x="Result", y="WeightedScore", data=df, ax=axes[0][1], palette=["#e74c3c", "#2ecc71"])
    axes[0][1].set_xticklabels(["Fail", "Pass"])
    axes[0][1].set_title("Weighted Score Distribution by Result")

    # 3. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0][2], linewidths=0.5)
    axes[0][2].set_title("Feature Correlation Heatmap")

    # 4. Feature Importance
    colors = ["#e74c3c" if v < 0 else "#2980b9" for v in imp_df[label]]
    axes[1][0].barh(imp_df["Feature"], imp_df[label], color=colors)
    axes[1][0].set_title("Feature Importance")
    axes[1][0].set_xlabel(label)
    axes[1][0].axvline(0, color="black", linewidth=0.8)

    # 5. Probability distribution — tuned threshold visualization
    axes[1][1].hist(probs[y_test.values == 0], bins=15, alpha=0.6, color="#e74c3c", label="Actual Fail")
    axes[1][1].hist(probs[y_test.values == 1], bins=15, alpha=0.6, color="#2ecc71", label="Actual Pass")
    axes[1][1].axvline(thresh, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={thresh}")
    axes[1][1].set_title("Prediction Probability Distribution")
    axes[1][1].set_xlabel("Predicted Probability of Passing")
    axes[1][1].legend()

    # 6. Attendance band pass/fail grouped bar
    band_counts = df.groupby(
        ["AttendanceBand", df["Result"].map({1: "Pass", 0: "Fail"})]
    ).size().unstack(fill_value=0)
    band_counts.plot(kind="bar", ax=axes[1][2], color=["#e74c3c", "#2ecc71"])
    axes[1][2].set_title("Pass/Fail Count by Attendance Band\n→ Action: Flag 'Low' band for intervention")
    axes[1][2].set_xlabel("Attendance Band")
    axes[1][2].tick_params(axis="x", rotation=0)
    axes[1][2].legend(title="Result")

    plt.tight_layout()
    plt.show()

def predict_student(results, best_name):
    print("\n─── PREDICT NEW STUDENT ───")
    attendance = float(input("Attendance %: "))
    maths = float(input("Maths marks: "))
    science = float(input("Science marks: "))
    english = float(input("English marks: "))

    total = maths + science + english
    average = total / 3
    weighted = (maths * 0.4) + (science * 0.3) + (english * 0.3)
    att_x_avg = attendance * average
    subject_std = np.std([maths, science, english])

    new_student = pd.DataFrame([[attendance, maths, science, english, total, average, weighted, att_x_avg, subject_std]],
        columns=["Attendance", "Maths", "Science", "English", "Total", "Average", "WeightedScore", "AttXAvg", "SubjectStd"])

    model = results[best_name]["model"]
    thresh = results[best_name]["best_threshold"]
    prob = model.predict_proba(new_student)[0]
    prediction = int(prob[1] >= thresh)

    print(f"\nResult     : {'✅ Pass' if prediction == 1 else '❌ Fail'}")
    print(f"Confidence → Pass: {prob[1]*100:.1f}%  |  Fail: {prob[0]*100:.1f}%")
    print(f"Threshold used: {thresh} (optimized for at-risk detection)")

    if prediction == 0:
        print("⚠️  HIGH RISK — Immediate counselor intervention recommended.")
        if attendance < 60:
            print("   → Primary driver: Critically low attendance.")
        if average < 40:
            print("   → Primary driver: Below-minimum academic performance.")
    elif prediction == 1 and prob[1] < 0.65:
        print("⚠️  BORDERLINE PASS — Monitor closely. Small dip could cause failure.")
    else:
        print("✅ Low risk. Student is on track.")

def print_decision_summary(df):
    fail = df[df["Result"] == 0]
    pass_ = df[df["Result"] == 1]
    low_att = df[df["Attendance"] < 60]
    low_att_fail_rate = len(low_att[low_att["Result"] == 0]) / len(low_att) * 100 if len(low_att) > 0 else 0
    att_corr = abs(df["Attendance"].corr(df["Result"]))
    avg_corr = abs(df["Average"].corr(df["Result"]))

    print("\n─── DECISION SUMMARY FOR SCHOOL ADMINISTRATION ───")
    print(f"1. Students below 60% attendance face a {low_att_fail_rate:.1f}% failure rate.")
    print(f"   → ACTION: Mandatory counseling for all students below 60% attendance threshold.")
    print(f"2. {'Attendance' if att_corr > avg_corr else 'Academic Score'} is the stronger failure predictor (r={max(att_corr, avg_corr):.2f}).")
    print(f"   → ACTION: Prioritize attendance monitoring over remedial classes.")
    print(f"3. Average marks gap between fail/pass: {pass_['Average'].mean() - fail['Average'].mean():.1f} points.")
    print(f"   → ACTION: Students scoring below {fail['Average'].mean() + 5:.0f} average should receive academic support.")
    print(f"\nThis system enables proactive intervention BEFORE exams — not reactive analysis after failure.")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

df = load_and_prepare("data/student_data.csv")
print("Dataset loaded. Shape:", df.shape)
print(df.head())

show_insights(df)

feature_cols = ["Attendance", "Maths", "Science", "English", "Total", "Average", "WeightedScore", "AttXAvg", "SubjectStd"]
X = df[feature_cols]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results, best_name = train_and_compare(X_train, X_test, y_train, y_test)

results = tune_threshold(results, best_name, X_test, y_test, feature_cols)

imp_df, label = show_feature_importance(results, best_name, feature_cols)

error_analysis(X_test, y_test, results, best_name)

visualize(df, imp_df, label, results, best_name, y_test)

print_decision_summary(df)

predict_student(results, best_name)