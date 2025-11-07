import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import tempfile
import os

# ----------------------------
# ⚙️ App Configuration
# ----------------------------
st.set_page_config(
    page_title="Glaucoma Severity Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# 🧠 Helper: Estimate CDR
# ----------------------------
def estimate_cdr(image):
    """Estimate Cup–Disc Ratio (CDR) and classify severity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    try:
        # Optic disc segmentation
        thresh_disc = threshold_otsu(gray)
        disc_mask = gray > thresh_disc * 0.9
        disc_mask = morphology.remove_small_objects(disc_mask, 400)
    except Exception:
        return None, None, "⚠️ No optic disc detected."

    if disc_mask.sum() == 0:
        return None, None, "⚠️ No optic disc detected."

    try:
        # Cup detection
        bright_thresh = np.percentile(gray[disc_mask], 80)
        cup_mask = gray > bright_thresh
        cup_mask = morphology.remove_small_objects(cup_mask, 100)
    except Exception:
        return None, None, "⚠️ No cup detected."

    if cup_mask.sum() == 0:
        return None, None, "⚠️ No cup detected."
    
    overlay = image.copy()
    overlay[disc_mask] = [0, 255, 0]     # Green for disc
    overlay[cup_mask] = [255, 0, 0]      # Red for cup


    # Calculate CDR
    disc_area = np.sum(disc_mask)
    cup_area = np.sum(cup_mask)
    cdr = min(cup_area / disc_area, 1.0)

    
    # --- Severity grading ---
    if cdr < 0.3:
        severity = "Normal Eye"
    elif cdr < 0.5:
        severity = "Mild Glaucoma"
    elif cdr < 0.7:
        severity = "Moderate Glaucoma"
    else:
        severity = "Severe Glaucoma"


    if severity == "Normal Eye":
        st.success(f"🟢 Result: {severity} (CDR: {cdr:.2f})")
    elif severity == "Mild Glaucoma":
        st.warning(f"🟡 Result: {severity} (CDR: {cdr:.2f})")
    elif severity == "Moderate Glaucoma":
        st.error(f"🟠 Result: {severity} (CDR: {cdr:.2f})")
    else:
        st.error(f"🔴 Result: {severity} (CDR: {cdr:.2f})")


# ----------------------------
# 🎨 Sidebar UI
# ----------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Eye_icon.svg", width=80)
st.sidebar.title("Glaucoma Analyzer 🩺")
st.sidebar.markdown("A simple automated tool for **glaucoma severity grading** using the Cup–Disc Ratio (CDR).")

theme_choice = st.sidebar.radio("🎨 Theme Mode", ["Light", "Dark"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("Developed by **Sai Dhanush**\nCapstone Project 2025")

# Apply dark mode styling (optional)
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
            body { background-color: #0e1117; color: white; }
            .stApp { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# 🌄 Main App Interface
# ----------------------------
st.title("🧠 Glaucoma Severity Grading System")
st.markdown(
    """
    Upload one or more **fundus images** to estimate the **Cup–Disc Ratio (CDR)** and determine **severity** of glaucoma.
    """
)

uploaded_files = st.file_uploader(
    "📤 Upload Fundus Image(s)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []

    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        cdr, severity, error = estimate_cdr(image)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                     caption=file.name, use_container_width=True)
        with col2:
            if error:
                st.error(f"{file.name}: {error}")
            else:
                st.success(f"**{file.name}** → CDR: `{cdr:.2f}` → Severity: **{severity}**")
                results.append({"Image": file.name, "CDR": round(cdr, 3), "Severity": severity})

    # Show results summary
    if results:
        df = pd.DataFrame(results)
        st.markdown("### 📊 Summary of Results")
        st.dataframe(df, use_container_width=True)

        # Save to temporary CSV
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(temp_csv.name, index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=open(temp_csv.name, "rb").read(),
            file_name="glaucoma_results.csv",
            mime="text/csv"
        )

        # Visualization section
        st.markdown("### 📈 Data Visualizations")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.hist(df["CDR"], bins=10, color="skyblue", edgecolor="black")
            ax.set_xlabel("Cup-to-Disc Ratio")
            ax.set_ylabel("Frequency")
            ax.set_title("CDR Distribution")
            st.pyplot(fig)

        with col2:
            st.bar_chart(df["Severity"].value_counts())

        avg_cdr = df["CDR"].mean()
        st.info(f"📏 **Average CDR:** {avg_cdr:.2f}")

else:
    st.info("👆 Upload one or more fundus images to get started.")




