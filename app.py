import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Student Risk Analyzer",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0e0;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #2ecc71;
        padding-bottom: 4px;
    }
    .risk-high {
        background: #3d1a1a;
        border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: #1a3d1a;
        border-left: 4px solid #2ecc71;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .risk-medium {
        background: #3d3200;
        border-left: 4px solid #f39c12;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .finding-box {
        background: #1a1f2e;
        border: 1px solid #2980b9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def prepare_data(df):
    df = df.copy()
    df["Result"] = df["Result"].map({"Pass": 1, "Fail": 0})
    df["Total"] = df["Maths"] + df["Science"] + df["English"]
    df["Average"] = df["Total"] / 3
    df["WeightedScore"] = (df["Maths"] * 0.4) + (df["Science"] * 0.3) + (df["English"] * 0.3)
    df["AttXAvg"] = df["Attendance"] * df["Average"]
    df["SubjectStd"] = df[["Maths", "Science", "English"]].std(axis=1)
    df["AttendanceBand"] = pd.cut(df["Attendance"], bins=[0, 60, 75, 100], labels=["Low", "Medium", "High"])
    df["PerformanceBand"] = pd.cut(df["Average"], bins=[0, 40, 60, 100], labels=["Weak", "Average", "Strong"])
    return df

def train_models(df):
    feature_cols = ["Attendance", "Maths", "Science", "English", "Total", "Average", "WeightedScore", "AttXAvg", "SubjectStd"]
    X = df[feature_cols]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        probs = model.predict_proba(X_test)[:, 1]

        best_thresh = 0.5
        best_recall = 0
        for thresh in [0.3, 0.35, 0.4, 0.45, 0.5]:
            custom_pred = (probs >= thresh).astype(int)
            report = classification_report(y_test, custom_pred, output_dict=True, zero_division=0)
            recall_fail = report.get("0", {}).get("recall", 0)
            if recall_fail > best_recall:
                best_recall = recall_fail
                best_thresh = thresh

        tuned_pred = (probs >= best_thresh).astype(int)
        report_dict = classification_report(y_test, tuned_pred, output_dict=True, zero_division=0)

        results[name] = {
            "model": model,
            "pred": tuned_pred,
            "probs": probs,
            "cv_f1": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "acc": model.score(X_test, y_test),
            "best_threshold": best_thresh,
            "report": report_dict,
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols
        }

    best_name = max(results, key=lambda k: results[k]["cv_f1"])
    return results, best_name, X_test, y_test

def get_feature_importance(results, best_name):
    model = results[best_name]["model"]
    feature_cols = results[best_name]["feature_cols"]
    if best_name == "Random Forest":
        imp = model.feature_importances_
        label = "Importance Score"
    else:
        imp = model.coef_[0]
        label = "Coefficient"
    imp_df = pd.DataFrame({"Feature": feature_cols, label: imp}).sort_values(label, ascending=False)
    return imp_df, label

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0e1117")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("🎓 Student Risk\nAnalyzer")
    st.markdown("---")
    st.markdown("**Upload your student CSV to begin.**")
    st.markdown("Required columns:")
    st.code("Attendance, Maths, Science,\nEnglish, Result (Pass/Fail)")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("Go to", ["📊 Overview", "🔬 Model Comparison", "🎯 Predict Student", "📥 Download Report"], label_visibility="collapsed")

# ─────────────────────────────────────────────
# LOAD + TRAIN
# ─────────────────────────────────────────────

if uploaded_file is None:
    st.markdown("<h1 style='text-align:center; color:#2ecc71; margin-top:5rem;'>🎓 Student Risk Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa; font-size:1.1rem;'>Upload a student CSV from the sidebar to begin analysis.</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='max-width:500px; margin:2rem auto; background:#1e2130; border-radius:12px; padding:1.5rem;'>
    <b style='color:#e0e0e0;'>What this app does:</b><br><br>
    ✅ Analyzes key failure drivers<br>
    ✅ Compares Logistic Regression vs Random Forest<br>
    ✅ Tunes threshold for at-risk detection<br>
    ✅ Predicts individual student risk with confidence<br>
    ✅ Generates downloadable HTML report
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Cache training so it doesn't rerun on every interaction
@st.cache_data
def load_and_train(file_bytes):
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df = prepare_data(df_raw)
    results, best_name, X_test, y_test = train_models(df)
    return df, results, best_name, X_test, y_test

file_bytes = uploaded_file.read()
with st.spinner("Training models and analyzing data..."):
    df, results, best_name, X_test, y_test = load_and_train(file_bytes)

fail = df[df["Result"] == 0]
pass_ = df[df["Result"] == 1]
low_att = df[df["Attendance"] < 60]
low_att_fail_rate = len(low_att[low_att["Result"] == 0]) / len(low_att) * 100 if len(low_att) > 0 else 0
att_corr = abs(df["Attendance"].corr(df["Result"]))
avg_corr = abs(df["Average"].corr(df["Result"]))
stronger_predictor = "Attendance" if att_corr > avg_corr else "Average Score"

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────

if page == "📊 Overview":
    st.markdown("<div class='section-title'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(df))
    c2.metric("Pass Rate", f"{df['Result'].mean()*100:.1f}%")
    c3.metric("Avg Attendance (Fail)", f"{fail['Attendance'].mean():.1f}%")
    c4.metric("Avg Attendance (Pass)", f"{pass_['Attendance'].mean():.1f}%")

    st.markdown("<div class='section-title'>📌 Hard Findings</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='finding-box'>
    📌 <b>Finding 1:</b> Students with &lt;60% attendance have a <b>{low_att_fail_rate:.1f}% failure rate</b> ({len(low_att[low_att['Result']==0])}/{len(low_att)} students).<br>
    → <i>ACTION: Mandatory counseling for all students below 60% attendance.</i>
    </div>
    <div class='finding-box'>
    📌 <b>Finding 2:</b> <b>'{stronger_predictor}'</b> is the stronger predictor of passing (r={max(att_corr, avg_corr):.2f}).<br>
    → <i>ACTION: Prioritize {"attendance monitoring" if stronger_predictor == "Attendance" else "academic support"} over other interventions.</i>
    </div>
    <div class='finding-box'>
    📌 <b>Finding 3:</b> Average marks gap between Fail and Pass students: <b>{pass_['Average'].mean() - fail['Average'].mean():.1f} points</b>.<br>
    → <i>ACTION: Flag students scoring below {fail['Average'].mean() + 5:.0f} average for academic support.</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📈 Visualizations</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        sns.boxplot(x="Result", y="Attendance", data=df, palette=["#e74c3c", "#2ecc71"], ax=ax)
        ax.set_xticklabels(["Fail", "Pass"], color="white")
        ax.set_title("Attendance vs Result", color="white")
        ax.set_xlabel("Result", color="#aaa")
        ax.set_ylabel("Attendance %", color="#aaa")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        sns.boxplot(x="Result", y="WeightedScore", data=df, palette=["#e74c3c", "#2ecc71"], ax=ax)
        ax.set_xticklabels(["Fail", "Pass"], color="white")
        ax.set_title("Weighted Score vs Result", color="white")
        ax.set_xlabel("Result", color="#aaa")
        ax.set_ylabel("Weighted Score", color="#aaa")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                    linewidths=0.5, annot_kws={"size": 7})
        ax.set_title("Feature Correlation Heatmap", color="white")
        ax.tick_params(colors="white", labelsize=7)
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        band_counts = df.groupby(
            ["AttendanceBand", df["Result"].map({1: "Pass", 0: "Fail"})]
        ).size().unstack(fill_value=0)
        band_counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
        ax.set_title("Pass/Fail by Attendance Band", color="white")
        ax.set_xlabel("Attendance Band", color="#aaa")
        ax.tick_params(colors="white", axis="both")
        ax.tick_params(axis="x", rotation=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(title="Result", labelcolor="white", title_fontsize=9)
        ax.title.set_color("white")
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-title'>📋 Attendance Band Breakdown</div>", unsafe_allow_html=True)
    crosstab = pd.crosstab(df["AttendanceBand"], df["Result"].map({1: "Pass", 0: "Fail"}), margins=True)
    st.dataframe(crosstab, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────

elif page == "🔬 Model Comparison":
    st.markdown("<div class='section-title'>🔬 Model Comparison Dashboard</div>", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            is_best = name == best_name
            border_color = "#2ecc71" if is_best else "#555"
            st.markdown(f"""
            <div style='border:2px solid {border_color}; border-radius:10px; padding:1rem; background:#1e2130;'>
            <h4 style='color:#e0e0e0;'>{"✅ " if is_best else ""}{name} {"← Best" if is_best else ""}</h4>
            </div>
            """, unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Test Accuracy", f"{res['acc']:.4f}")
            m2.metric("CV F1", f"{res['cv_f1']:.4f}")
            m3.metric("Threshold", f"{res['best_threshold']}")

            report = res["report"]
            report_df = pd.DataFrame({
                "Class": ["Fail", "Pass"],
                "Precision": [report.get("0", {}).get("precision", 0), report.get("1", {}).get("precision", 0)],
                "Recall": [report.get("0", {}).get("recall", 0), report.get("1", {}).get("recall", 0)],
                "F1-Score": [report.get("0", {}).get("f1-score", 0), report.get("1", {}).get("f1-score", 0)],
            }).set_index("Class")
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    st.markdown("<div class='section-title'>📊 Feature Importance</div>", unsafe_allow_html=True)
    imp_df, label = get_feature_importance(results, best_name)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1e2130")
    ax.set_facecolor("#1e2130")
    colors = ["#e74c3c" if v < 0 else "#2980b9" for v in imp_df[label]]
    ax.barh(imp_df["Feature"], imp_df[label], color=colors)
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_title(f"Feature Importance — {best_name}", color="white")
    ax.set_xlabel(label, color="#aaa")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig)
    plt.close()

    st.markdown("<div class='section-title'>📉 Threshold Tuning Table</div>", unsafe_allow_html=True)
    model = results[best_name]["model"]
    probs = results[best_name]["probs"]
    y_test_vals = results[best_name]["y_test"]

    thresh_rows = []
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5]:
        custom_pred = (probs >= thresh).astype(int)
        r = classification_report(y_test_vals, custom_pred, output_dict=True, zero_division=0)
        thresh_rows.append({
            "Threshold": thresh,
            "Recall (Fail)": round(r.get("0", {}).get("recall", 0), 4),
            "Precision (Fail)": round(r.get("0", {}).get("precision", 0), 4),
            "F1 (Fail)": round(r.get("0", {}).get("f1-score", 0), 4),
        })
    thresh_df = pd.DataFrame(thresh_rows).set_index("Threshold")
    st.dataframe(thresh_df.style.highlight_max(color="#1a3d1a", axis=0), use_container_width=True)

    st.markdown("<div class='section-title'>📊 Probability Distribution</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#1e2130")
    ax.set_facecolor("#1e2130")
    y_vals = y_test_vals.values if hasattr(y_test_vals, "values") else y_test_vals
    ax.hist(probs[y_vals == 0], bins=15, alpha=0.6, color="#e74c3c", label="Actual Fail")
    ax.hist(probs[y_vals == 1], bins=15, alpha=0.6, color="#2ecc71", label="Actual Pass")
    ax.axvline(results[best_name]["best_threshold"], color="white", linestyle="--", linewidth=1.5,
               label=f"Threshold={results[best_name]['best_threshold']}")
    ax.set_title("Prediction Probability Distribution", color="white")
    ax.set_xlabel("Predicted Probability of Passing", color="#aaa")
    ax.tick_params(colors="white")
    ax.legend(labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
# PAGE: PREDICT STUDENT
# ─────────────────────────────────────────────

elif page == "🎯 Predict Student":
    st.markdown("<div class='section-title'>🎯 Predict Individual Student Risk</div>", unsafe_allow_html=True)
    st.markdown("Enter the student's details below. The model will predict pass/fail probability and flag risk level.")

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance %", 0, 100, 75)
        maths = st.slider("Maths Marks", 0, 100, 60)
    with col2:
        science = st.slider("Science Marks", 0, 100, 60)
        english = st.slider("English Marks", 0, 100, 60)

    if st.button("🔍 Predict", use_container_width=True):
        total = maths + science + english
        average = total / 3
        weighted = (maths * 0.4) + (science * 0.3) + (english * 0.3)
        att_x_avg = attendance * average
        subject_std = np.std([maths, science, english])

        new_student = pd.DataFrame(
            [[attendance, maths, science, english, total, average, weighted, att_x_avg, subject_std]],
            columns=["Attendance", "Maths", "Science", "English", "Total", "Average", "WeightedScore", "AttXAvg", "SubjectStd"]
        )

        model = results[best_name]["model"]
        thresh = results[best_name]["best_threshold"]
        prob = model.predict_proba(new_student)[0]
        prediction = int(prob[1] >= thresh)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", "✅ Pass" if prediction == 1 else "❌ Fail")
        r2.metric("Pass Confidence", f"{prob[1]*100:.1f}%")
        r3.metric("Fail Confidence", f"{prob[0]*100:.1f}%")

        if prediction == 0:
            reasons = []
            if attendance < 60:
                reasons.append("Critically low attendance (&lt;60%)")
            if average < 40:
                reasons.append("Below-minimum average score (&lt;40)")
            if subject_std > 20:
                reasons.append("High inconsistency across subjects")
            reason_html = "".join([f"<br>• {r}" for r in reasons]) if reasons else "<br>• Borderline performance across indicators"
            st.markdown(f"<div class='risk-high'>⚠️ <b>HIGH RISK</b> — Immediate counselor intervention recommended.{reason_html}</div>", unsafe_allow_html=True)
        elif prediction == 1 and prob[1] < 0.65:
            st.markdown("<div class='risk-medium'>⚠️ <b>BORDERLINE PASS</b> — Monitor closely. Small dip in attendance or marks could cause failure.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-low'>✅ <b>LOW RISK</b> — Student is on track. Continue monitoring.</div>", unsafe_allow_html=True)

        st.markdown(f"<p style='color:#888; font-size:0.85rem;'>Model: {best_name} | Threshold: {thresh} (optimized for at-risk recall)</p>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(5, 2), facecolor="#1e2130")
        ax.set_facecolor("#1e2130")
        ax.barh(["Fail", "Pass"], [prob[0]*100, prob[1]*100], color=["#e74c3c", "#2ecc71"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence %", color="#aaa")
        ax.set_title("Prediction Confidence", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────
# PAGE: DOWNLOAD REPORT
# ─────────────────────────────────────────────

elif page == "📥 Download Report":
    st.markdown("<div class='section-title'>📥 Generate Downloadable Report</div>", unsafe_allow_html=True)
    st.markdown("Click below to generate a full HTML report with insights, model comparison, and action recommendations.")

    if st.button("⚙️ Generate Report", use_container_width=True):
        with st.spinner("Building report..."):
            imp_df, label = get_feature_importance(results, best_name)
            lr_res = results["Logistic Regression"]
            rf_res = results["Random Forest"]

            html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>Student Risk Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0e1117; color: #e0e0e0; margin: 0; padding: 2rem; }}
  h1 {{ color: #2ecc71; border-bottom: 2px solid #2ecc71; padding-bottom: 0.5rem; }}
  h2 {{ color: #81ecec; margin-top: 2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th {{ background: #1e2130; color: #2ecc71; padding: 0.6rem 1rem; text-align: left; }}
  td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #2a2f45; }}
  tr:hover {{ background: #1a1f2e; }}
  .finding {{ background: #1a1f2e; border-left: 4px solid #2980b9; padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 4px; }}
  .action {{ background: #1a2e1a; border-left: 4px solid #2ecc71; padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 4px; }}
  .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 2rem; }}
  .badge {{ background: #2ecc71; color: #0e1117; padding: 2px 10px; border-radius: 20px; font-weight: bold; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>🎓 Student Risk Analysis Report</h1>
<p class='meta'>Dataset: {len(df)} students | Best Model: <span class='badge'>{best_name}</span> | Generated by Student Risk Analyzer</p>

<h2>📊 Dataset Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Students</td><td>{len(df)}</td></tr>
  <tr><td>Pass Rate</td><td>{df['Result'].mean()*100:.1f}%</td></tr>
  <tr><td>Fail Rate</td><td>{(1-df['Result'].mean())*100:.1f}%</td></tr>
  <tr><td>Avg Attendance (Fail)</td><td>{fail['Attendance'].mean():.1f}%</td></tr>
  <tr><td>Avg Attendance (Pass)</td><td>{pass_['Attendance'].mean():.1f}%</td></tr>
  <tr><td>Avg Score (Fail)</td><td>{fail['Average'].mean():.1f}</td></tr>
  <tr><td>Avg Score (Pass)</td><td>{pass_['Average'].mean():.1f}</td></tr>
</table>

<h2>📌 Key Findings</h2>
<div class='finding'>📌 Students with &lt;60% attendance have a <b>{low_att_fail_rate:.1f}%</b> failure rate.</div>
<div class='finding'>📌 <b>'{stronger_predictor}'</b> is the stronger predictor of passing (r={max(att_corr, avg_corr):.2f}).</div>
<div class='finding'>📌 Average marks gap between Fail and Pass: <b>{pass_['Average'].mean() - fail['Average'].mean():.1f} points</b>.</div>

<h2>🔬 Model Comparison</h2>
<table>
  <tr><th>Model</th><th>Test Accuracy</th><th>CV F1 (5-fold)</th><th>Optimal Threshold</th></tr>
  <tr><td>Logistic Regression</td><td>{lr_res['acc']:.4f}</td><td>{lr_res['cv_f1']:.4f} ± {lr_res['cv_std']:.4f}</td><td>{lr_res['best_threshold']}</td></tr>
  <tr><td>Random Forest</td><td>{rf_res['acc']:.4f}</td><td>{rf_res['cv_f1']:.4f} ± {rf_res['cv_std']:.4f}</td><td>{rf_res['best_threshold']}</td></tr>
</table>
<p>✅ Best model selected: <b>{best_name}</b> (highest CV F1 score)</p>

<h2>📊 Feature Importance ({best_name})</h2>
<table>
  <tr><th>Feature</th><th>{label}</th></tr>
  {"".join(f"<tr><td>{row['Feature']}</td><td>{row[label]:.4f}</td></tr>" for _, row in imp_df.iterrows())}
</table>

<h2>🎯 Action Recommendations</h2>
<div class='action'>✅ <b>Action 1:</b> Mandatory counseling for all students below 60% attendance threshold.</div>
<div class='action'>✅ <b>Action 2:</b> Prioritize {"attendance monitoring" if stronger_predictor == "Attendance" else "academic support"} as the primary intervention lever.</div>
<div class='action'>✅ <b>Action 3:</b> Flag students scoring below {fail['Average'].mean() + 5:.0f} average for academic support programs.</div>
<div class='action'>✅ <b>Action 4:</b> Use threshold={results[best_name]['best_threshold']} in the prediction system to maximize detection of at-risk students.</div>

</body>
</html>
"""

        b64 = base64.b64encode(html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="student_risk_report.html" style="display:inline-block; background:#2ecc71; color:#0e1117; padding:0.7rem 1.5rem; border-radius:8px; text-decoration:none; font-weight:bold; margin-top:1rem;">📥 Download HTML Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("Report ready. Click the button above to download.")