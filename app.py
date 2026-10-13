import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Campus Complaint Priority Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Campus Complaint Priority Prediction")
st.caption("NLP + Machine Learning | Student complaint triage demo")

DATA_FILE = Path(__file__).parent / "campus_complaints_100.csv"

@st.cache_data
def load_data(path, modified_time):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

if not DATA_FILE.exists():
    st.error("Dataset file not found. Keep campus_complaints_100.csv in the same folder as app.py.")
    st.stop()

df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)

required = {"Complaint", "Priority"}
if not required.issubset(df.columns):
    st.error("CSV must contain these exact columns: Complaint, Priority")
    st.write("Columns found:", list(df.columns))
    st.stop()

df = df.dropna(subset=["Complaint", "Priority"]).copy()
df["Complaint"] = df["Complaint"].astype(str).str.strip()
df["Priority"] = df["Priority"].astype(str).str.strip().str.title()
df = df[(df["Complaint"] != "") & (df["Priority"] != "")]

allowed = {"Urgent", "High", "Normal", "Low"}
unknown = sorted(set(df["Priority"]) - allowed)
if unknown:
    st.warning(f"Priority labels outside the expected four categories: {unknown}")

if df["Priority"].nunique() < 2:
    st.error("At least two different priority labels are needed to train a classifier.")
    st.stop()

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
model.fit(df["Complaint"], df["Priority"])

priority_order = ["Urgent", "High", "Normal", "Low"]
counts = df["Priority"].value_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total complaints", len(df))
c2.metric("Urgent", int(counts.get("Urgent", 0)))
c3.metric("High", int(counts.get("High", 0)))
c4.metric("Normal", int(counts.get("Normal", 0)))
c5.metric("Low", int(counts.get("Low", 0)))

tab_predict, tab_manage, tab_evaluate = st.tabs(["🔎 Predict Priority", "📋 Complaint Dataset", "📊 Model Check"])

with tab_predict:
    st.subheader("Analyse a new complaint")
    complaint = st.text_area(
        "Enter complaint text",
        placeholder="Example: Water leakage in hostel",
        height=100
    )
    if st.button("Predict priority", type="primary"):
        if not complaint.strip():
            st.warning("Please enter a complaint first.")
        else:
            prediction = model.predict([complaint.strip()])[0]
            st.success(f"Predicted priority: **{prediction.upper()}**")
            st.info("This is a model prediction, not a guaranteed decision. Review safety-related complaints manually.")
            if hasattr(model.named_steps["classifier"], "predict_proba"):
                probs = model.predict_proba([complaint.strip()])[0]
                prob_df = pd.DataFrame({
                    "Priority": model.named_steps["classifier"].classes_,
                    "Model score": probs
                }).sort_values("Model score", ascending=False)
                st.write("Model scores (not guaranteed real-world probabilities):")
                st.dataframe(prob_df.style.format({"Model score": "{:.1%}"}), use_container_width=True)

with tab_manage:
    st.subheader("Browse complaints")
    selected = st.multiselect(
        "Filter by priority",
        options=priority_order,
        default=[p for p in priority_order if p in set(df["Priority"])]
    )
    search = st.text_input("Search complaint text")
    filtered = df[df["Priority"].isin(selected)]
    if search.strip():
        filtered = filtered[filtered["Complaint"].str.contains(search.strip(), case=False, na=False)]
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
    st.download_button(
        "Download displayed rows as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_campus_complaints.csv",
        mime="text/csv"
    )

with tab_evaluate:
    st.subheader("Basic model check")
    st.warning(
        "This dataset has only 100 sample rows and was created for demonstration. "
        "Training accuracy is shown only as a quick check; it is not a reliable estimate of performance on new complaints."
    )
    training_predictions = model.predict(df["Complaint"])
    st.metric("Training accuracy (not test accuracy)", f"{accuracy_score(df['Priority'], training_predictions):.1%}")
    st.text("Training classification report:")
    st.code(classification_report(df["Priority"], training_predictions, zero_division=0))
    st.caption(
        "For a fair project evaluation, use a separate held-out test set and add more diverse, correctly labelled complaints. "
        "Urgent safety complaints should always be reviewed by a human."
    )

st.divider()
st.caption("Campus Complaint Priority Predictor • Demo software")
