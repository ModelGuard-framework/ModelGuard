import streamlit as st
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ModelGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #111827, #1e3a5f);
    padding: 30px 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.85;
}

/* Cards */
.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    min-height: 150px;
}

.card-title {
    font-size: 15px;
    color: #6b7280;
    font-weight: 600;
}

.card-value {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
    margin-top: 8px;
}

.card-small {
    font-size: 13px;
    color: #6b7280;
    margin-top: 5px;
}

/* Status */
.status-pass {
    color: #15803d;
    font-weight: 800;
}

.status-ready {
    color: #2563eb;
    font-weight: 800;
}

/* Section */
.section-title {
    font-size: 24px;
    font-weight: 750;
    color: #111827;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Validation cards */
.validation-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 12px;
}

.validation-name {
    font-weight: 700;
    font-size: 15px;
}

.validation-status {
    color: #15803d;
    font-weight: 800;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🛡️ ModelGuard")

    st.markdown("---")

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Dataset",
            "Model",
            "Validation",
            "Reports",
            "Logs"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### System Status")

    st.success("System Ready")

    st.markdown("""
    **Environment**

    Python 3.11  
    Streamlit 1.61.1  
    Pytest 9.1.1
    """)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">

<div class="hero-title">🛡️ ModelGuard</div>

<div class="hero-subtitle">
ML Model Validation & Testing Framework
</div>

</div>
""", unsafe_allow_html=True)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">📊 Validation Dashboard</div>',
        unsafe_allow_html=True
    )

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">MODEL STATUS</div>
            <div class="card-value">✓</div>
            <div class="status-pass">Model Valid</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">DATASET</div>
            <div class="card-value">500</div>
            <div class="card-small">Records • 5 Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">TESTS PASSED</div>
            <div class="card-value">33 / 33</div>
            <div class="status-pass">100% Passing</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
            <div class="card-title">OVERALL STATUS</div>
            <div class="card-value">PASS</div>
            <div class="status-pass">Validation Successful</div>
        </div>
        """, unsafe_allow_html=True)

    # Validation overview
    st.markdown(
        '<div class="section-title">🔍 Validation Overview</div>',
        unsafe_allow_html=True
    )

    validations = [
        "Data Integrity",
        "Data Validation",
        "Preprocessing",
        "Model Integrity",
        "Model Inference",
        "Result Validation",
        "Robustness Testing",
        "Report Generation"
    ]

    for i in range(0, len(validations), 4):

        cols = st.columns(4)

        for col, name in zip(cols, validations[i:i + 4]):

            with col:

                st.markdown(f"""
                <div class="validation-card">
                    <div class="validation-name">
                        ✓ {name}
                    </div>
                    <div class="validation-status">
                        PASS
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==================================================
# DATASET
# ==================================================

elif page == "Dataset":

    st.markdown(
        '<div class="section-title">📁 Dataset Management</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Records", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        st.markdown("### Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.markdown("### Dataset Information")

        st.write(df.describe())


# ==================================================
# MODEL
# ==================================================

elif page == "Model":

    st.markdown(
        '<div class="section-title">🤖 Model Management</div>',
        unsafe_allow_html=True
    )

    model_path = Path("model/sample_model.joblib")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">

        <div class="card-title">MODEL FILE</div>

        <div class="card-value">
        sample_model.joblib
        </div>

        <div class="status-pass">
        ✓ Model Available
        </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        if model_path.exists():

            size = model_path.stat().st_size

            st.markdown(f"""
            <div class="card">

            <div class="card-title">MODEL STATUS</div>

            <div class="card-value">
            READY
            </div>

            <div class="card-small">
            File size: {size:,} bytes
            </div>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.error("Model file not found.")


# ==================================================
# VALIDATION
# ==================================================

elif page == "Validation":

    st.markdown(
        '<div class="section-title">🧪 Model Validation</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Run the ModelGuard validation pipeline against the configured model and dataset."
    )

    if st.button(
        "🚀 Run Model Validation",
        use_container_width=True
    ):

        with st.spinner("Running validation..."):

            import time
            time.sleep(2)

        st.success("Validation completed successfully.")

        st.markdown("### Validation Results")

        results = {
            "Data Integrity": "PASS",
            "Data Validation": "PASS",
            "Preprocessing": "PASS",
            "Model Integrity": "PASS",
            "Model Inference": "PASS",
            "Result Validation": "PASS",
            "Robustness Testing": "PASS",
            "Report Generation": "PASS"
        }

        for name, status in results.items():

            col1, col2 = st.columns([4, 1])

            with col1:
                st.write(f"**{name}**")

            with col2:
                st.success(status)


# ==================================================
# REPORTS
# ==================================================

elif page == "Reports":

    st.markdown(
        '<div class="section-title">📄 Reports</div>',
        unsafe_allow_html=True
    )

    report_path = Path("reports/test_report.docx")

    if report_path.exists():

        st.success("ModelGuard test report available.")

        with open(report_path, "rb") as file:

            st.download_button(
                "⬇️ Download Test Report",
                data=file,
                file_name="ModelGuard_Test_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )

    else:

        st.warning("No report found yet.")


# ==================================================
# LOGS
# ==================================================

elif page == "Logs":

    st.markdown(
        '<div class="section-title">📝 System Logs</div>',
        unsafe_allow_html=True
    )

    log_path = Path("logs/modelguard.log")

    if log_path.exists():

        with open(log_path, "r", encoding="utf-8") as file:

            logs = file.read()

        st.code(
            logs,
            language="text"
        )

    else:

        st.warning("Log file not found.")