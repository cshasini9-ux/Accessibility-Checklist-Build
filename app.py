"""
app.py
------
AI Accessibility Checklist Builder
A Streamlit web app that evaluates a digital product's accessibility,
generates a score, classifies it using a Random Forest model, gives
improvement suggestions, and lets the user download a report.

Run with:
    streamlit run app.py
"""

import io
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from model_utils import (
    calculate_accessibility_score,
    train_model,
    predict_category,
    generate_recommendations,
)

# ----------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Accessibility Checklist Builder",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Simple custom CSS for a cleaner, more modern look
# ----------------------------------------------------------------------
st.markdown("""
    <style>
        .main { background-color: #f8f9fb; }
        .score-card {
            padding: 25px;
            border-radius: 14px;
            text-align: center;
            color: white;
            font-weight: 600;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5em 1.2em;
        }
        .reco-box {
            background-color: #ffffff;
            border-left: 5px solid #4C8BF5;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cache the trained model so it's only trained once per session
# ----------------------------------------------------------------------
@st.cache_resource
def get_trained_model():
    df = pd.read_csv("sample_dataset.csv")
    model, label_encoder, accuracy = train_model(df)
    return model, label_encoder, accuracy


model, label_encoder, model_accuracy = get_trained_model()

# ----------------------------------------------------------------------
# Session state setup (keeps results available across pages)
# ----------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("♿ Accessibility Checklist Builder")
st.sidebar.markdown("AI-powered UI/UX accessibility evaluation tool")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📝 Assessment Form", "📊 Dashboard", "📄 Download Report"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"AI Model Accuracy on test data: **{model_accuracy * 100:.1f}%**")
st.sidebar.caption("Built with Streamlit, Scikit-learn, Pandas & Matplotlib")


# ----------------------------------------------------------------------
# PAGE 1: HOME
# ----------------------------------------------------------------------
if page == "🏠 Home":
    st.title("✅ AI Accessibility Checklist Builder")
    st.subheader("Evaluate. Score. Improve. Make your product accessible to everyone.")

    st.write(
        """
        This tool helps UI/UX teams quickly evaluate the **accessibility**
        of a digital product (website, app, or prototype) against common
        WCAG-style criteria. It calculates an accessibility score,
        classifies the design using a trained **AI model (Random Forest)**,
        and provides actionable recommendations.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nFill out the Assessment Form with details about your product's accessibility features.")
    with col2:
        st.info("**Step 2**\n\nView your Score, AI Classification, and Charts on the Dashboard.")
    with col3:
        st.info("**Step 3**\n\nDownload a complete accessibility report to share with your team.")

    st.markdown("---")
    st.write("👉 Use the sidebar to get started with the **Assessment Form**.")


# ----------------------------------------------------------------------
# PAGE 2: ASSESSMENT FORM
# ----------------------------------------------------------------------
elif page == "📝 Assessment Form":
    st.title("📝 Accessibility Assessment Form")
    st.write("Answer the questions below about your digital product.")

    with st.form("assessment_form"):
        col1, col2 = st.columns(2)

        with col1:
            product_name = st.text_input("Product / Page Name", value="My Website")

            font_size = st.slider(
                "Font Size (px)", min_value=8, max_value=32, value=14,
                help="WCAG recommends a minimum body text size of 16px."
            )

            color_contrast = st.slider(
                "Color Contrast Ratio", min_value=1.0, max_value=21.0, value=3.5, step=0.1,
                help="WCAG AA requires a minimum contrast ratio of 4.5:1 for normal text."
            )

            alt_text = st.selectbox(
                "Image Alt Text Availability",
                ["Yes", "Partial", "No"],
                help="Do all images have descriptive alternative text?"
            )

        with col2:
            keyboard_nav = st.selectbox(
                "Keyboard Navigation Support",
                ["Yes", "No"],
                help="Can users navigate the entire interface using only a keyboard?"
            )

            screen_reader = st.selectbox(
                "Screen Reader Compatibility",
                ["Yes", "Partial", "No"],
                help="Is the content correctly read out loud by screen readers (e.g. NVDA, VoiceOver)?"
            )

            form_labels = st.selectbox(
                "Form Label Availability",
                ["Yes", "Partial", "No"],
                help="Do all input fields have clear, associated labels?"
            )

        submitted = st.form_submit_button("🔍 Evaluate Accessibility")

    if submitted:
        # --- 1. Rule-based score ---
        breakdown, total_score = calculate_accessibility_score(
            font_size, color_contrast, alt_text, keyboard_nav, screen_reader, form_labels
        )
        status = "PASS ✅" if total_score >= 60 else "FAIL ❌"

        # --- 2. AI classification ---
        ai_category = predict_category(
            model, label_encoder, font_size, color_contrast,
            alt_text, keyboard_nav, screen_reader, form_labels
        )

        # --- 3. Recommendations ---
        recommendations = generate_recommendations(
            font_size, color_contrast, alt_text, keyboard_nav, screen_reader, form_labels
        )

        # Save everything to session state so other pages can use it
        st.session_state.results = {
            "product_name": product_name,
            "font_size": font_size,
            "color_contrast": color_contrast,
            "alt_text": alt_text,
            "keyboard_nav": keyboard_nav,
            "screen_reader": screen_reader,
            "form_labels": form_labels,
            "breakdown": breakdown,
            "total_score": total_score,
            "status": status,
            "ai_category": ai_category,
            "recommendations": recommendations,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        st.success("Evaluation complete! Head over to the 📊 Dashboard to view your results.")


# ----------------------------------------------------------------------
# PAGE 3: DASHBOARD
# ----------------------------------------------------------------------
elif page == "📊 Dashboard":
    st.title("📊 Accessibility Dashboard")

    if st.session_state.results is None:
        st.warning("No assessment found yet. Please fill out the 📝 Assessment Form first.")
    else:
        r = st.session_state.results

        # --- Score card + AI classification ---
        score = r["total_score"]
        color = "#2ecc71" if score >= 85 else ("#f39c12" if score >= 60 else "#e74c3c")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""<div class="score-card" style="background-color:{color};">
                <h2>{score}/100</h2><p>Accessibility Score</p></div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="score-card" style="background-color:#34495e;">
                <h2>{r['status']}</h2><p>Pass / Fail Status</p></div>""",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""<div class="score-card" style="background-color:#4C8BF5;">
                <h2>{r['ai_category']}</h2><p>AI Classification</p></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Charts ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Accessibility Factors (Points Earned)")
            breakdown = r["breakdown"]
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            bars = ax1.bar(breakdown.keys(), breakdown.values(), color="#4C8BF5")
            ax1.set_ylabel("Points")
            ax1.set_ylim(0, 22)
            ax1.set_xticklabels(breakdown.keys(), rotation=30, ha="right")
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f"{height}", (bar.get_x() + bar.get_width() / 2, height),
                             ha="center", va="bottom", fontsize=8)
            fig1.tight_layout()
            st.pyplot(fig1)

        with chart_col2:
            st.subheader("Overall Compliance")
            compliant = score
            non_compliant = 100 - score
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.pie(
                [compliant, non_compliant],
                labels=["Compliant", "Non-Compliant"],
                autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"],
                startangle=90,
            )
            ax2.axis("equal")
            st.pyplot(fig2)

        st.markdown("---")

        # --- Recommendations ---
        st.subheader("💡 Automated Recommendations")
        for tip in r["recommendations"]:
            st.markdown(f"<div class='reco-box'>🔸 {tip}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# PAGE 4: DOWNLOAD REPORT
# ----------------------------------------------------------------------
elif page == "📄 Download Report":
    st.title("📄 Download Accessibility Report")

    if st.session_state.results is None:
        st.warning("No assessment found yet. Please fill out the 📝 Assessment Form first.")
    else:
        r = st.session_state.results

        report_lines = [
            "AI ACCESSIBILITY CHECKLIST BUILDER - EVALUATION REPORT",
            "=" * 55,
            f"Product / Page Name : {r['product_name']}",
            f"Generated On        : {r['timestamp']}",
            "",
            "INPUT DETAILS",
            "-" * 55,
            f"Font Size                     : {r['font_size']} px",
            f"Color Contrast Ratio           : {r['color_contrast']}:1",
            f"Image Alt Text Availability    : {r['alt_text']}",
            f"Keyboard Navigation Support    : {r['keyboard_nav']}",
            f"Screen Reader Compatibility    : {r['screen_reader']}",
            f"Form Label Availability        : {r['form_labels']}",
            "",
            "SCORE BREAKDOWN",
            "-" * 55,
        ]
        for factor, points in r["breakdown"].items():
            report_lines.append(f"{factor:<28}: {points} points")

        report_lines += [
            "",
            f"TOTAL ACCESSIBILITY SCORE      : {r['total_score']} / 100",
            f"PASS / FAIL STATUS             : {r['status']}",
            f"AI CLASSIFICATION              : {r['ai_category']}",
            "",
            "RECOMMENDATIONS",
            "-" * 55,
        ]
        for i, tip in enumerate(r["recommendations"], start=1):
            report_lines.append(f"{i}. {tip}")

        report_text = "\n".join(report_lines)

        st.text_area("Report Preview", report_text, height=400)

        report_bytes = io.BytesIO(report_text.encode("utf-8"))

        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report_bytes,
            file_name=f"accessibility_report_{r['product_name'].replace(' ', '_')}.txt",
            mime="text/plain",
        )
