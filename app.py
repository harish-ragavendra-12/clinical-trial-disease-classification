import streamlit as st

from src.prediction import prediction_pipeline


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Clinical Trial Disease Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
GOOGLE FONT
========================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html,
body,
[class*="css"]{

    font-family:'Inter',sans-serif;

}


/* ==========================================================
BACKGROUND
========================================================== */

.stApp{

    background:#F5F9FC;

}


/* ==========================================================
HIDE STREAMLIT
========================================================== */

#MainMenu{

    visibility:hidden;

}

footer{

    visibility:hidden;

}

header{

    visibility:hidden;

}


/* ==========================================================
MAIN PAGE TEXT
========================================================== */

h1,
h2,
h3,
h4,
h5,
h6{

    color:#0F172A;

}

p,
label,
li,
span{

    color:#334155;

}


/* ==========================================================
SIDEBAR
========================================================== */

[data-testid="stSidebar"]{

    background:linear-gradient(
        180deg,
        #0F172A,
        #1E3A8A,
        #2563EB
    );

}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li{

    color:white;

}


/* ==========================================================
BUTTONS
========================================================== */

.stButton>button{

    width:100%;

    height:55px;

    border-radius:14px;

    border:none;

    background:linear-gradient(
        90deg,
        #2563EB,
        #1D4ED8
    );

    color:white;

    font-size:17px;

    font-weight:700;

    transition:.25s;

    box-shadow:
        0px 6px 18px rgba(37,99,235,.20);

}

.stButton>button:hover{

    background:#1D4ED8;

    transform:translateY(-2px);

}


/* ==========================================================
TEXT AREA
========================================================== */

textarea{

    border-radius:16px !important;

    border:2px solid #CBD5E1 !important;

    background:white !important;

    color:#0F172A !important;

    font-size:16px !important;

}


/* ==========================================================
INPUT
========================================================== */

input{

    color:#0F172A !important;

}


/* ==========================================================
CONTAINERS
========================================================== */

[data-testid="stVerticalBlock"]{

    color:#0F172A;

}


/* ==========================================================
METRICS
========================================================== */

[data-testid="metric-container"]{

    background:white;

    border-radius:18px;

    border:1px solid #E2E8F0;

    padding:18px;

    box-shadow:
        0px 8px 20px rgba(0,0,0,.06);

}

[data-testid="metric-container"] label{

    color:#64748B !important;

}

[data-testid="metric-container"] div{

    color:#0F172A !important;

}


/* ==========================================================
SUCCESS
========================================================== */

[data-testid="stSuccess"]{

    border-radius:14px;

}


/* ==========================================================
INFO
========================================================== */

[data-testid="stInfo"]{

    border-radius:14px;

}


/* ==========================================================
WARNING
========================================================== */

[data-testid="stWarning"]{

    border-radius:14px;

}


/* ==========================================================
EXPANDER
========================================================== */

.streamlit-expanderHeader{

    color:#0F172A;

    font-weight:700;

}


/* ==========================================================
DOWNLOAD BUTTON
========================================================== */

.stDownloadButton>button{

    width:100%;

    height:55px;

    border-radius:14px;

}


/* ==========================================================
DIVIDER
========================================================== */

hr{

    border:1px solid #E2E8F0;

}

/* ==========================================================
HERO BANNER
========================================================== */

.hero-title{
    color:#FFFFFF !important;
    font-size:52px;
    font-weight:800;
    margin-bottom:0;
}

.hero-subtitle{
    color:#E2E8F0 !important;
    font-size:20px;
    font-weight:500;
    margin-top:12px;
}

/* ==========================================================
FOOTER
========================================================== */

.footer-title{
    color:#FFFFFF !important;
    font-size:42px;
    font-weight:800;
}

.footer-subtitle{
    color:#E2E8F0 !important;
    font-size:20px;
}

.footer-tech{
    color:#FFFFFF !important;
    font-size:18px;
}

.footer-copy{
    color:#CBD5E1 !important;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("# 🧬 Clinical Trial AI")

    st.caption("Healthcare Analytics Dashboard")

    st.divider()

    st.subheader("📖 Project")

    st.write("""
Predict disease categories from
clinical trial summaries using
Natural Language Processing
and Machine Learning.
""")

    st.divider()

    st.subheader("⚙️ Technologies")

    st.markdown("""
- 🐍 Python

- 🤖 Scikit-Learn

- 📄 TF-IDF

- 🎈 Streamlit

- 💾 Joblib
""")

    st.divider()

    st.subheader("🔬 Workflow")

    st.markdown("""
Clinical Trial Summary

⬇️

Text Cleaning

⬇️

TF-IDF Vectorization

⬇️

Machine Learning

⬇️

Disease Prediction
""")

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.write("Harish Ragavendra")

    st.caption("Data Science Portfolio Project")


# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div style="
background:linear-gradient(90deg,#0F172A,#1E3A8A,#2563EB);
padding:35px;
border-radius:20px;
box-shadow:0px 12px 30px rgba(0,0,0,.18);
">

<div class="hero-title">
🧬 Clinical Trial Disease Classification
</div>

<div class="hero-subtitle">
AI Powered Disease Prediction using Natural Language Processing & Machine Learning
</div>

</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.write("")

with st.container(border=True):

    st.subheader("📖 About This Project")

    st.write("""
This application predicts the disease category of a
clinical trial from its summary using
Natural Language Processing (NLP)
and Machine Learning.

The prediction pipeline consists of:

• Text Cleaning

• TF-IDF Feature Engineering

• Machine Learning Classification

• Disease Prediction

This project demonstrates a complete end-to-end
Machine Learning workflow from preprocessing
to deployment using Streamlit.
""")


# ==========================================================
# SESSION STATE
# ==========================================================

if "input_text" not in st.session_state:

    st.session_state.input_text = ""

if "predicted_disease" not in st.session_state:

    st.session_state.predicted_disease = None


# ==========================================================
# SAMPLE CLINICAL TRIAL
# ==========================================================

example_text = """
A randomized clinical trial evaluating the effectiveness
of insulin therapy in adult patients diagnosed with
type 2 diabetes mellitus.

The study compares insulin therapy with oral medication
while monitoring HbA1c, fasting blood glucose,
adverse effects and treatment safety.
"""


# ==========================================================
# INPUT SECTION
# ==========================================================

st.write("")

with st.container(border=True):

    st.subheader("📝 Clinical Trial Summary")

    st.caption(
        "Enter the clinical trial summary below and let the AI model predict the disease category."
    )

    left_button, right_button = st.columns(2)

    with left_button:

        if st.button(
            "📄 Load Example",
            use_container_width=True
        ):

            st.session_state.input_text = example_text

            st.session_state.predicted_disease = None

            st.rerun()

    with right_button:

        if st.button(
            "🗑 Clear",
            use_container_width=True
        ):

            st.session_state.input_text = ""

            st.session_state.predicted_disease = None

            st.rerun()

    st.write("")

    input_text = st.text_area(

        label="Clinical Trial Summary",

        value=st.session_state.input_text,

        height=280,

        placeholder="""
Example

A randomized clinical trial evaluating insulin therapy
in patients diagnosed with Type 2 Diabetes Mellitus.
The study measures treatment effectiveness,
blood glucose level, HbA1c,
adverse events and overall safety...
"""
    )

    st.session_state.input_text = input_text

    st.write("")

    predict_clicked = st.button(

        "🔍 Predict Disease",

        use_container_width=True
    )


# ==========================================================
# PREDICTION
# ==========================================================

if predict_clicked:

    if input_text.strip() == "":

        st.warning(
            "⚠ Please enter a clinical trial summary."
        )

    else:

        with st.spinner(
            "🧠 AI is analyzing the clinical trial..."
        ):

            try:

                predicted_disease = prediction_pipeline(
                    input_text
                )

                st.session_state.predicted_disease = predicted_disease

            except Exception as error:

                st.error(
                    f"Prediction Failed\n\n{error}"
                )

# ==========================================================
# PREDICTION RESULT
# ==========================================================

if st.session_state.predicted_disease is not None:

    st.write("")
    st.subheader("🩺 Prediction Result")
    st.write("")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            label="Predicted Disease",
            value=st.session_state.predicted_disease
        )

    with metric2:

        st.metric(
            label="Model Status",
            value="Loaded"
        )

    with metric3:

        st.metric(
            label="Prediction",
            value="Completed"
        )

    st.write("")

    # ======================================================
    # Prediction Card
    # ======================================================

    st.markdown(
        f"""
    <div style="background:linear-gradient(90deg,#0F766E,#0D9488,#14B8A6);padding:30px;border-radius:18px;color:white;box-shadow:0px 12px 30px rgba(0,0,0,.18);">

    <h2 style="margin:0;color:white;">
    🧬 Predicted Disease
    </h2>

    <h1 style="margin-top:15px;font-size:42px;color:white;font-weight:700;">
    {st.session_state.predicted_disease}
    </h1>

    <p style="margin-top:15px;font-size:18px;color:#ECFDF5;">
    Prediction generated using the trained Machine Learning model.
    </p>

    </div>
    """,
        unsafe_allow_html=True
    )

    st.write("")

    st.info("""
    This prediction is generated using the trained Machine Learning model.

    The prediction is intended for educational purposes and should not replace professional medical advice.
    """)

    st.write("")

    # ======================================================
    # Prediction Summary
    # ======================================================

    with st.expander(
        "📄 View Prediction Details",
        expanded=True
    ):

        st.markdown("### Clinical Trial Summary")

        st.write(
            st.session_state.input_text
        )

        st.divider()

        st.markdown("### Predicted Disease")

        st.success(
            st.session_state.predicted_disease
        )

        st.divider()

        st.markdown("### Machine Learning Pipeline")

        st.markdown("""

✅ Text Cleaning

✅ TF-IDF Vectorization

✅ Machine Learning Prediction

✅ Label Decoding

""")

    st.write("")

    # ======================================================
    # Download Report
    # ======================================================

    report = f"""
Clinical Trial Disease Classification Report

====================================================

Predicted Disease

{st.session_state.predicted_disease}

====================================================

Clinical Trial Summary

{st.session_state.input_text}

====================================================

Generated Using

Python

Scikit-Learn

TF-IDF

Streamlit

====================================================
"""

    st.download_button(

        label="📥 Download Prediction Report",

        data=report,

        file_name="prediction_report.txt",

        mime="text/plain",

        use_container_width=True

    )



# ==========================================================
# HOW IT WORKS
# ==========================================================

st.write("")
st.divider()

with st.expander(
    "⚙️ How the Prediction Works",
    expanded=False
):

    st.markdown("""

### 🧠 Machine Learning Workflow

1. User enters a clinical trial summary.

2. Text preprocessing removes unwanted characters.

3. TF-IDF converts text into numerical vectors.

4. The trained Machine Learning model predicts the disease.

5. The encoded label is converted back into the original disease name.

---

### Models Evaluated

• Logistic Regression

• Multinomial Naive Bayes

• Random Forest

• Linear Support Vector Classifier

The best performing model is automatically
selected during training.

""")

# ==========================================================
# PROJECT HIGHLIGHTS
# ==========================================================

st.write("")
st.subheader("🚀 Project Highlights")

left_col, right_col = st.columns(2)

with left_col:

    with st.container(border=True):

        st.markdown("### 🤖 Machine Learning")

        st.markdown("""

✅ Natural Language Processing

✅ TF-IDF Feature Engineering

✅ Multiple Classification Models

✅ Automatic Best Model Selection

✅ Model Serialization using Joblib

""")


with right_col:

    with st.container(border=True):

        st.markdown("### 💻 Application")

        st.markdown("""

✅ Interactive Streamlit Dashboard

✅ Professional Healthcare UI

✅ Clinical Trial Prediction

✅ Download Prediction Report

✅ End-to-End ML Deployment

""")


# ==========================================================
# TECHNOLOGIES
# ==========================================================

st.write("")
st.subheader("⚙️ Technology Stack")

tech1, tech2, tech3, tech4, tech5 = st.columns(5)

with tech1:
    st.info("🐍\n\nPython")

with tech2:
    st.info("🤖\n\nScikit-Learn")

with tech3:
    st.info("📄\n\nTF-IDF")

with tech4:
    st.info("🎈\n\nStreamlit")

with tech5:
    st.info("💾\n\nJoblib")


# ==========================================================
# TIPS
# ==========================================================

st.write("")

with st.expander("💡 Tips for Better Predictions"):

    st.markdown("""

- Enter complete clinical trial summaries.

- Mention treatments or interventions.

- Include disease names if available.

- Avoid very short descriptions.

- Provide meaningful clinical context.

""")


# ==========================================================
# ABOUT DEVELOPER
# ==========================================================

st.write("")
st.subheader("👨‍💻 About the Developer")

with st.container(border=True):

    st.markdown("""

### Harish Ragavendra

Aspiring **Data Scientist** passionate about:

- Machine Learning

- Natural Language Processing

- Healthcare Analytics

- End-to-End AI Applications

This project demonstrates the complete Machine Learning
lifecycle including:

- Data Cleaning

- Feature Engineering

- Model Training

- Model Evaluation

- Prediction Pipeline

- Streamlit Deployment

""")


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown("""
<div style="
background:linear-gradient(90deg,#0F172A,#1E3A8A,#2563EB);
padding:40px;
border-radius:20px;
text-align:center;
margin-top:40px;
">

<div class="footer-title">
🧬 Clinical Trial Disease Classification
</div>

<div class="footer-subtitle">
AI Powered Disease Prediction using NLP & Machine Learning
</div>

<hr style="
margin:30px 0;
border:1px solid rgba(255,255,255,.25);
">

<div class="footer-tech">
🐍 Python &nbsp; | &nbsp;
🤖 Scikit-Learn &nbsp; | &nbsp;
📄 TF-IDF &nbsp; | &nbsp;
📍 Streamlit
</div>

<br>

<div class="footer-copy">
© 2026 Harish Ragavendra
</div>

</div>
""", unsafe_allow_html=True)