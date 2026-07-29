import streamlit as st

from src.prediction import prediction_pipeline


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Clinical Trial Disease Classification",
    page_icon="🧬",
    layout="wide"
)


# ==========================================================
# Title
# ==========================================================

st.title("🧬 Clinical Trial Disease Classification")
st.markdown(
    "Predict the disease category of a clinical trial using **Natural Language Processing (NLP)** and **Machine Learning**."
)

st.divider()


# ==========================================================
# About
# ==========================================================

with st.expander("📖 About This Project", expanded=True):

    st.write("""
This application predicts the disease category of a clinical trial
based on its summary.

### Machine Learning Pipeline

- Data Cleaning
- TF-IDF Feature Engineering
- Model Prediction
- Disease Classification
""")


# ==========================================================
# Input Section
# ==========================================================

st.subheader("📝 Clinical Trial Summary")

input_text = st.text_area(
    "Enter the clinical trial summary:",
    height=250,
    placeholder="Paste the clinical trial summary here..."
)

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button(
        "🔍 Predict Disease",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear_button:
    st.rerun()


# ==========================================================
# Prediction
# ==========================================================

if predict_button:

    if input_text.strip() == "":

        st.warning("Please enter a clinical trial summary.")

    else:

        with st.spinner("Predicting disease..."):

            try:

                predicted_disease = prediction_pipeline(input_text)

                st.success("Prediction Completed Successfully!")

                st.subheader("🩺 Predicted Disease")

                st.info(predicted_disease)

            except Exception as e:

                st.error(f"Prediction Failed\n\n{e}")


st.divider()


# ==========================================================
# Footer
# ==========================================================

st.caption(
    "Developed using Python • Scikit-Learn • TF-IDF • Streamlit"
)