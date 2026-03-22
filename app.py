from importlib import import_module
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from inference.predict_freight import predict_freight_cost


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
FREIGHT_MODEL_PATH = MODELS_DIR / "predict_freight_model.pkl"
INVOICE_MODEL_PATH = MODELS_DIR / "predict_flag_invoice.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
INVOICE_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

invoice_flag_module = import_module("inference.predict_invoice_flag")

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_freight_model():
    return joblib.load(FREIGHT_MODEL_PATH)


@st.cache_resource
def load_invoice_assets():
    model = joblib.load(INVOICE_MODEL_PATH)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def run_freight_prediction(dollars: float) -> float:
    input_data = {"Dollars": [float(dollars)]}

    try:
        prediction_df = predict_freight_cost(input_data)
    except Exception:
        model = load_freight_model()
        prediction_df = pd.DataFrame(input_data)
        prediction_df["Predicted_Freight"] = model.predict(
            prediction_df[["Dollars"]]
        ).round(2)

    return float(prediction_df["Predicted_Freight"].iloc[0])


def run_invoice_flag_prediction(input_data: dict) -> int:
    if hasattr(invoice_flag_module, "predict_invoice_flag"):
        try:
            result_df = invoice_flag_module.predict_invoice_flag(input_data)
            return int(result_df["Predicted_Flag"].iloc[0])
        except Exception:
            pass

    model, scaler = load_invoice_assets()
    input_df = pd.DataFrame(input_data)
    scaled_features = scaler.transform(input_df[INVOICE_FEATURES])
    predictions = model.predict(scaled_features)
    return int(predictions[0])


def render_result_card(label: str, value: str, tone: str = "default"):
    st.markdown(
        f"""
        <div class="result-card result-card-{tone}">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_asset_check_panel():
    assets = [
        ("Freight model", FREIGHT_MODEL_PATH),
        ("Invoice flag model", INVOICE_MODEL_PATH),
        ("Feature scaler", SCALER_PATH),
    ]
    ready_count = sum(path.exists() for _, path in assets)
    total_count = len(assets)
    overall_status = "All systems ready" if ready_count == total_count else "Attention needed"

    rows = []
    for label, path in assets:
        status_class = "ready" if path.exists() else "missing"
        status_text = "Ready" if path.exists() else "Missing"
        rows.append(
            (
                f'<div class="asset-row">'
                f'<div>'
                f'<div class="asset-name">{label}</div>'
                f'<div class="asset-path">{path.relative_to(ROOT_DIR)}</div>'
                f"</div>"
                f'<div class="asset-badge asset-badge-{status_class}">'
                f'<span class="asset-dot asset-dot-{status_class}"></span>'
                f"{status_text}"
                f"</div>"
                f"</div>"
            )
        )

    st.markdown(
        (
            f'<div class="asset-panel">'
            f'<div class="asset-panel-top">'
            f'<div>'
            f'<div class="asset-title">Model Asset Status</div>'
            f'<div class="asset-subtitle">{overall_status}</div>'
            f"</div>"
            f'<div class="asset-count">{ready_count}/{total_count}</div>'
            f"</div>"
            f'<div class="asset-list">{"".join(rows)}</div>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #07131d;
                --paper: rgba(10, 22, 34, 0.96);
                --panel-bg: rgba(12, 28, 42, 0.88);
                --panel-strong: rgba(14, 33, 49, 0.96);
                --panel-border: rgba(120, 159, 189, 0.18);
                --text-strong: #ecf4fb;
                --text-muted: #9fb4c7;
                --accent-deep: #11998e;
                --accent-mid: #38bdf8;
                --accent-bright: #8b5cf6;
                --accent-soft: rgba(56, 189, 248, 0.16);
                --olive: #7fd4ca;
                --sidebar-bg-top: #050b11;
                --sidebar-bg-bottom: #091521;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(17, 153, 142, 0.18), transparent 26%),
                    radial-gradient(circle at top right, rgba(139, 92, 246, 0.16), transparent 28%),
                    linear-gradient(180deg, var(--app-bg) 0%, #091723 48%, #050c14 100%);
                color: var(--text-strong);
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1220px;
            }

            header[data-testid="stHeader"] {
                background: rgba(7, 19, 29, 0.78);
                backdrop-filter: blur(10px);
            }

            [data-testid="stToolbar"] {
                right: 1rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(
                    180deg,
                    var(--sidebar-bg-top) 0%,
                    var(--sidebar-bg-bottom) 100%
                ) !important;
                border-right: 1px solid rgba(120, 159, 189, 0.12);
                box-shadow: 12px 0 30px rgba(0, 0, 0, 0.28);
            }

            [data-testid="stSidebar"] > div:first-child {
                background: linear-gradient(
                    180deg,
                    var(--sidebar-bg-top) 0%,
                    var(--sidebar-bg-bottom) 100%
                ) !important;
            }

            [data-testid="stSidebarContent"] {
                padding-top: 1.25rem;
            }

            [data-testid="stSidebar"] * {
                color: #dfeaf4 !important;
            }

            [data-testid="stSidebar"] .stRadio > label,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] .stCaption,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] li,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label {
                color: #dfeaf4 !important;
            }

            [data-testid="stSidebar"] .stRadio > div {
                background: rgba(56, 189, 248, 0.08);
                border: 1px solid rgba(120, 159, 189, 0.14);
                border-radius: 20px;
                padding: 0.7rem 0.8rem;
            }

            [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
                border-radius: 14px;
                padding: 0.42rem 0.22rem;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #f5fbff !important;
                letter-spacing: -0.02em;
            }

            [data-testid="stSidebar"] .stCaption {
                color: rgba(159, 180, 199, 0.84) !important;
            }

            div[data-testid="stForm"] {
                background: linear-gradient(180deg, rgba(12, 28, 42, 0.96) 0%, rgba(9, 22, 34, 0.98) 100%);
                border: 1px solid var(--panel-border);
                border-radius: 28px;
                padding: 1.1rem 1.1rem 0.5rem 1.1rem;
                box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
            }

            [data-testid="stNumberInput"] label,
            [data-testid="stTextInput"] label,
            [data-testid="stSelectbox"] label {
                color: var(--text-strong) !important;
                font-weight: 600;
            }

            [data-testid="stNumberInput"] input,
            [data-testid="stTextInput"] input,
            [data-baseweb="select"] > div {
                background: rgba(6, 16, 25, 0.98) !important;
                color: var(--text-strong) !important;
                border-radius: 14px !important;
                border: 1px solid rgba(120, 159, 189, 0.2) !important;
            }

            [data-testid="stNumberInput"] button {
                color: var(--text-strong) !important;
            }

            [data-testid="stNumberInput"] input::placeholder,
            [data-testid="stTextInput"] input::placeholder {
                color: rgba(159, 180, 199, 0.72) !important;
            }

            .hero-card {
                background:
                    radial-gradient(circle at top right, rgba(139, 92, 246, 0.24), transparent 22%),
                    radial-gradient(circle at bottom left, rgba(17, 153, 142, 0.2), transparent 24%),
                    linear-gradient(135deg, #0a1722 0%, #10263a 48%, #0b1a27 100%);
                border: 1px solid rgba(120, 159, 189, 0.14);
                border-radius: 30px;
                color: #f4fbff;
                padding: 2rem;
                box-shadow: 0 28px 72px rgba(0, 0, 0, 0.32);
                margin-bottom: 1.4rem;
            }

            .hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.6fr) minmax(240px, 0.9fr);
                gap: 1rem;
                align-items: stretch;
            }

            .hero-copy {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .hero-kicker {
                display: inline-block;
                margin-bottom: 0.8rem;
                border-radius: 999px;
                background: rgba(56, 189, 248, 0.1);
                border: 1px solid rgba(120, 159, 189, 0.18);
                padding: 0.38rem 0.72rem;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(223, 234, 244, 0.9);
            }

            .hero-card h1 {
                margin: 0 0 0.7rem 0;
                font-size: 2.5rem;
                line-height: 1.08;
                letter-spacing: -0.03em;
            }

            .hero-card p {
                margin: 0;
                font-size: 1rem;
                color: rgba(223, 234, 244, 0.82);
                max-width: 760px;
            }

            .badge-row {
                display: flex;
                gap: 0.6rem;
                flex-wrap: wrap;
                margin-top: 1.15rem;
            }

            .badge {
                background: rgba(56, 189, 248, 0.08);
                border: 1px solid rgba(120, 159, 189, 0.18);
                border-radius: 999px;
                padding: 0.45rem 0.82rem;
                font-size: 0.88rem;
                color: #eaf6ff;
            }

            .hero-summary {
                background: linear-gradient(180deg, rgba(12, 31, 46, 0.86) 0%, rgba(8, 20, 31, 0.92) 100%);
                border: 1px solid rgba(120, 159, 189, 0.18);
                border-radius: 24px;
                padding: 1.15rem 1rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 0.9rem;
            }

            .hero-summary-label {
                color: rgba(159, 180, 199, 0.84);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .hero-summary-value {
                font-size: 2rem;
                font-weight: 700;
                line-height: 1;
                color: #f5fbff;
            }

            .hero-summary-list {
                display: grid;
                gap: 0.65rem;
                color: rgba(223, 234, 244, 0.82);
                font-size: 0.93rem;
            }

            .overview-card {
                min-height: 168px;
            }

            .eyebrow {
                margin-bottom: 0.42rem;
                color: #67d6f7;
                font-size: 0.76rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
            }

            .info-card {
                background: linear-gradient(180deg, rgba(12, 28, 42, 0.9) 0%, rgba(9, 21, 33, 0.96) 100%);
                border: 1px solid var(--panel-border);
                border-radius: 24px;
                padding: 1.15rem 1.2rem;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
                min-height: 152px;
            }

            .info-card h3 {
                margin: 0 0 0.45rem 0;
                font-size: 1.08rem;
                color: var(--text-strong);
            }

            .info-card p {
                margin: 0;
                color: var(--text-muted);
                line-height: 1.6;
                font-size: 0.95rem;
            }

            .section-title {
                margin-top: 0.45rem;
                margin-bottom: 0.2rem;
                color: var(--text-strong);
                font-size: 2.25rem;
                letter-spacing: -0.03em;
            }

            .section-copy {
                color: var(--text-muted);
                margin-bottom: 1rem;
                max-width: 860px;
                font-size: 1rem;
                line-height: 1.65;
            }

            .note-box {
                background: linear-gradient(135deg, rgba(17, 153, 142, 0.16) 0%, rgba(56, 189, 248, 0.08) 100%);
                border: 1px solid rgba(17, 153, 142, 0.2);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                color: #c7f7f0;
                margin-bottom: 1rem;
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(10, 26, 40, 0.96) 0%, rgba(6, 18, 29, 0.94) 100%);
                border: 1px solid rgba(120, 159, 189, 0.16);
                border-radius: 20px;
                padding: 0.9rem 1rem;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
            }

            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] [data-testid="stMetricLabel"],
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: var(--text-strong) !important;
            }

            .stButton > button,
            .stFormSubmitButton > button {
                width: 100%;
                border-radius: 999px;
                border: none;
                background: linear-gradient(135deg, var(--accent-deep) 0%, var(--accent-mid) 100%);
                color: #f5fbff;
                font-weight: 600;
                letter-spacing: 0.01em;
                padding: 0.74rem 1rem;
                box-shadow: 0 12px 24px rgba(17, 153, 142, 0.22);
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                background: linear-gradient(135deg, #0e7f77 0%, #2498cc 100%);
            }

            .module-aside {
                display: grid;
                gap: 1rem;
            }

            .signal-list {
                display: grid;
                gap: 0.75rem;
                margin-top: 0.78rem;
            }

            .signal-item {
                display: flex;
                gap: 0.7rem;
                align-items: flex-start;
            }

            .signal-dot {
                width: 0.72rem;
                height: 0.72rem;
                margin-top: 0.35rem;
                border-radius: 999px;
                background: linear-gradient(135deg, var(--accent-mid), var(--accent-bright));
                box-shadow: 0 0 0 6px rgba(56, 189, 248, 0.14);
                flex: 0 0 auto;
            }

            .asset-panel {
                margin-top: 0.3rem;
                background: linear-gradient(180deg, rgba(12, 28, 42, 0.82) 0%, rgba(7, 18, 28, 0.94) 100%);
                border: 1px solid rgba(120, 159, 189, 0.14);
                border-radius: 20px;
                padding: 0.9rem 0.9rem 0.8rem 0.9rem;
            }

            .asset-panel-top {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 0.8rem;
                margin-bottom: 0.8rem;
            }

            .asset-title {
                color: #f5fbff;
                font-size: 0.92rem;
                font-weight: 700;
                line-height: 1.3;
            }

            .asset-subtitle {
                color: rgba(159, 180, 199, 0.9);
                font-size: 0.82rem;
                margin-top: 0.18rem;
            }

            .asset-count {
                flex: 0 0 auto;
                min-width: 3rem;
                text-align: center;
                border-radius: 999px;
                padding: 0.28rem 0.58rem;
                background: rgba(56, 189, 248, 0.1);
                border: 1px solid rgba(120, 159, 189, 0.16);
                color: #dff6ff;
                font-size: 0.86rem;
                font-weight: 700;
            }

            .asset-list {
                display: grid;
                gap: 0.65rem;
            }

            .asset-row {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 0.7rem;
                padding: 0.72rem 0.78rem;
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(120, 159, 189, 0.08);
            }

            .asset-name {
                color: #edf7ff;
                font-size: 0.88rem;
                font-weight: 600;
                line-height: 1.3;
            }

            .asset-path {
                color: rgba(159, 180, 199, 0.88);
                font-size: 0.76rem;
                line-height: 1.4;
                margin-top: 0.16rem;
                overflow-wrap: anywhere;
            }

            .asset-badge {
                flex: 0 0 auto;
                display: inline-flex;
                align-items: center;
                gap: 0.38rem;
                border-radius: 999px;
                padding: 0.28rem 0.56rem;
                font-size: 0.76rem;
                font-weight: 700;
                line-height: 1;
            }

            .asset-badge-ready {
                background: rgba(17, 153, 142, 0.16);
                border: 1px solid rgba(17, 153, 142, 0.22);
                color: #c8faf3;
            }

            .asset-badge-missing {
                background: rgba(244, 63, 94, 0.14);
                border: 1px solid rgba(244, 63, 94, 0.2);
                color: #ffd7e1;
            }

            .asset-dot {
                width: 0.42rem;
                height: 0.42rem;
                border-radius: 999px;
                flex: 0 0 auto;
            }

            .asset-dot-ready {
                background: #34d399;
                box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.12);
            }

            .asset-dot-missing {
                background: #fb7185;
                box-shadow: 0 0 0 4px rgba(251, 113, 133, 0.12);
            }

            .result-card {
                background: linear-gradient(180deg, rgba(10, 26, 40, 0.96) 0%, rgba(6, 18, 29, 0.94) 100%);
                border: 1px solid rgba(120, 159, 189, 0.16);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
                min-height: 122px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .result-card-success {
                border-color: rgba(17, 153, 142, 0.26);
            }

            .result-card-warning {
                border-color: rgba(139, 92, 246, 0.3);
            }

            .result-label {
                color: var(--text-strong);
                font-size: 0.95rem;
                font-weight: 600;
                line-height: 1.35;
            }

            .result-value {
                color: #f7fbff;
                font-size: clamp(1.75rem, 2.6vw, 2.25rem);
                font-weight: 700;
                line-height: 1.12;
                letter-spacing: -0.03em;
                overflow-wrap: anywhere;
                word-break: break-word;
            }

            .result-meta {
                color: var(--text-muted);
                font-size: 0.95rem;
                line-height: 1.7;
                margin-top: 0.9rem;
            }

            .result-meta code {
                color: #7fe5d6;
                background: rgba(17, 153, 142, 0.12);
                border: 1px solid rgba(17, 153, 142, 0.18);
                border-radius: 8px;
                padding: 0.12rem 0.38rem;
            }

            .stSuccess,
            .stWarning,
            .stError,
            .stInfo {
                border-radius: 18px;
            }

            .stCaption,
            .stMarkdown,
            p,
            li,
            label {
                color: inherit;
            }

            @media (max-width: 900px) {
                .block-container {
                    padding-top: 1rem;
                }

                .hero-card {
                    padding: 1.5rem 1.25rem;
                    border-radius: 24px;
                }

                .hero-grid {
                    grid-template-columns: 1fr;
                }

                .hero-card h1 {
                    font-size: 1.92rem;
                }

                .info-card {
                    min-height: unset;
                }

                .section-title {
                    font-size: 1.9rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div>
                        <div class="hero-kicker">Finance Intelligence Workspace</div>
                        <h1>Vendor Invoice Intelligence Portal</h1>
                        <p>
                            A warmer, more editorial control room for freight forecasting and invoice risk screening.
                            Review likely shipping exposure, catch invoices that deserve a second look, and keep
                            finance operations moving with less guesswork.
                        </p>
                    </div>
                    <div class="badge-row">
                        <span class="badge">Freight estimate lane</span>
                        <span class="badge">Approval risk lane</span>
                        <span class="badge">Internal operations dashboard</span>
                    </div>
                </div>
                <div class="hero-summary">
                    <div>
                        <div class="hero-summary-label">Live Modules</div>
                        <div class="hero-summary-value">2</div>
                    </div>
                    <div class="hero-summary-list">
                        <div>Freight prediction from invoice value</div>
                        <div>Manual approval scoring from review features</div>
                        <div>Designed for finance and procurement teams</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_cards():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card overview-card">
                <div class="eyebrow">Operational Lens</div>
                <h3>Operations Focus</h3>
                <p>
                    Estimate freight cost early in the review cycle and keep vendor conversations
                    grounded in likely shipping exposure before payment decisions are finalized.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card overview-card">
                <div class="eyebrow">Review Lens</div>
                <h3>Finance Review</h3>
                <p>
                    Surface invoices that deserve a closer look when freight, quantities,
                    and total item values begin to drift away from expected patterns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="info-card overview-card">
                <div class="eyebrow">Model Lens</div>
                <h3>Deployment Notes</h3>
                <p>
                    The current deployment mixes a compact freight regression model with a
                    richer invoice risk classifier that reads five review-oriented features.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> str:
    with st.sidebar:
        st.title("Navigation")
        st.caption("Choose the workflow you want to review in this session.")
        selected_model = st.radio(
            "Prediction workspace",
            options=(
                "Freight Cost Prediction",
                "Invoice Manual Approval Flag",
            ),
        )

        st.markdown("---")
        st.markdown("### Why It Matters")
        st.markdown(
            """
            - Make freight exposure visible earlier in the approval process.
            - Route suspicious invoices to finance reviewers faster.
            - Keep low-risk invoices moving with a stronger first pass.
            """
        )

        st.markdown("### Asset Check")
        render_asset_check_panel()

    return selected_model


def render_freight_view():
    st.markdown('<h2 class="section-title">Freight Cost Prediction</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="section-copy">
            Enter invoice context to estimate freight cost from the deployed regression model.
            The currently saved model predicts from <strong>Invoice Dollars</strong>; quantity is
            captured as supporting business context for the operator.
        </p>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.35, 0.9])

    with left_col:
        with st.form("freight_form"):
            st.markdown(
                """
                <div class="note-box">
                    Use this view when procurement or finance teams want a quick freight estimate
                    before finalizing invoice handling.
                </div>
                """,
                unsafe_allow_html=True,
            )

            input_col1, input_col2 = st.columns(2)

            with input_col1:
                quantity = st.number_input(
                    "Invoice Quantity",
                    min_value=1,
                    value=1200,
                    step=1,
                    help="Captured for operator context in the dashboard.",
                )

            with input_col2:
                dollars = st.number_input(
                    "Invoice Dollars",
                    min_value=1.0,
                    value=18500.0,
                    step=100.0,
                    format="%.2f",
                    help="Primary input used by the currently deployed freight model.",
                )

            submit_freight = st.form_submit_button("Predict Freight Cost")

        if submit_freight:
            predicted_freight = run_freight_prediction(dollars)

            metric_col1, metric_col2, metric_col3 = st.columns([1.05, 1.15, 0.9])
            with metric_col1:
                render_result_card(
                    "Estimated Freight Cost",
                    f"${predicted_freight:,.2f}",
                    tone="success",
                )
            with metric_col2:
                render_result_card("Invoice Dollars", f"${dollars:,.2f}")
            with metric_col3:
                render_result_card("Invoice Quantity", f"{int(quantity):,}")

            st.success("Prediction completed successfully.")
            st.markdown(
                """
                <div class="result-meta">
                    Freight estimate generated from the deployed regression model stored in
                    <code>models/predict_freight_model.pkl</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown(
            """
            <div class="module-aside">
                <div class="info-card">
                    <div class="eyebrow">Where It Helps</div>
                    <h3>Recommended Use</h3>
                    <p>
                        Compare expected freight against vendor-submitted values during invoice review,
                        budgeting, or exception triage.
                    </p>
                </div>
                <div class="info-card">
                    <div class="eyebrow">Current Behavior</div>
                    <h3>Current Model Scope</h3>
                    <p>
                        Training code shows the saved freight model was fit on the <strong>Dollars</strong>
                        feature only. Quantity is shown in the UI for business context, not as a model feature.
                    </p>
                </div>
                <div class="info-card">
                    <div class="eyebrow">Signals To Watch</div>
                    <h3>What Teams Usually Compare</h3>
                    <div class="signal-list">
                        <div class="signal-item">
                            <span class="signal-dot"></span>
                            <p>Expected freight versus submitted freight charge</p>
                        </div>
                        <div class="signal-item">
                            <span class="signal-dot"></span>
                            <p>Large invoices with unusually small or large freight outcomes</p>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_invoice_flag_view():
    st.markdown(
        '<h2 class="section-title">Invoice Manual Approval Prediction</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="section-copy">
            Score a vendor invoice for manual approval review using the deployed classification model.
            This workflow combines invoice quantity, invoice dollars, freight, and purchase totals.
        </p>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.35, 0.9])

    with left_col:
        with st.form("invoice_flag_form"):
            st.markdown(
                """
                <div class="note-box">
                    Use realistic invoice and purchase totals. The model output is a decision-support
                    signal, not a replacement for final finance review.
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                invoice_quantity = st.number_input(
                    "Invoice Quantity",
                    min_value=1,
                    value=50,
                    step=1,
                )
                freight = st.number_input(
                    "Freight Cost",
                    min_value=0.0,
                    value=1.73,
                    step=0.1,
                    format="%.2f",
                )

            with col2:
                invoice_dollars = st.number_input(
                    "Invoice Dollars",
                    min_value=1.0,
                    value=352.95,
                    step=10.0,
                    format="%.2f",
                )
                total_item_quantity = st.number_input(
                    "Total Item Quantity",
                    min_value=1,
                    value=162,
                    step=1,
                )

            with col3:
                total_item_dollars = st.number_input(
                    "Total Item Dollars",
                    min_value=1.0,
                    value=2476.0,
                    step=10.0,
                    format="%.2f",
                )

            submit_flag = st.form_submit_button("Evaluate Invoice Risk")

        if submit_flag:
            input_data = {
                "invoice_quantity": [int(invoice_quantity)],
                "invoice_dollars": [float(invoice_dollars)],
                "Freight": [float(freight)],
                "total_item_quantity": [int(total_item_quantity)],
                "total_item_dollars": [float(total_item_dollars)],
            }

            flag_prediction = run_invoice_flag_prediction(input_data)
            needs_manual_review = bool(flag_prediction)

            status_col1, status_col2 = st.columns([1.15, 0.95])
            with status_col1:
                status_text = (
                    "Manual approval required"
                    if needs_manual_review
                    else "Safe for auto-approval"
                )
                render_result_card(
                    "Decision",
                    status_text,
                    tone="warning" if needs_manual_review else "success",
                )

            with status_col2:
                render_result_card(
                    "Invoice Amount Gap",
                    f"${abs(total_item_dollars - invoice_dollars):,.2f}",
                )

            if needs_manual_review:
                st.error(
                    "The model recommends routing this invoice to a manual approval workflow."
                )
            else:
                st.success(
                    "The model indicates this invoice is low risk for manual approval escalation."
                )

            st.markdown(
                """
                <div class="result-meta">
                    Classification result generated from the model in
                    <code>models/predict_flag_invoice.pkl</code>
                    with the saved scaler in <code>models/scaler.pkl</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown(
            """
            <div class="module-aside">
                <div class="info-card">
                    <div class="eyebrow">Escalation Logic</div>
                    <h3>Manual Review Signals</h3>
                    <p>
                        Large gaps between invoice dollars and aggregated item dollars can be a strong
                        signal that a review is worth triggering.
                    </p>
                </div>
                <div class="info-card">
                    <div class="eyebrow">Input Design</div>
                    <h3>Model Inputs</h3>
                    <p>
                        The deployed classifier uses five numeric fields:
                        invoice quantity, invoice dollars, freight, total item quantity,
                        and total item dollars.
                    </p>
                </div>
                <div class="info-card">
                    <div class="eyebrow">Analyst Habit</div>
                    <h3>Useful Review Pattern</h3>
                    <div class="signal-list">
                        <div class="signal-item">
                            <span class="signal-dot"></span>
                            <p>Check the invoice amount gap before approving exceptions</p>
                        </div>
                        <div class="signal-item">
                            <span class="signal-dot"></span>
                            <p>Use the model outcome as a triage layer, not as the final decision</p>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_missing_assets_warning():
    missing_assets = [
        str(path.relative_to(ROOT_DIR))
        for path in (FREIGHT_MODEL_PATH, INVOICE_MODEL_PATH, SCALER_PATH)
        if not path.exists()
    ]

    if missing_assets:
        st.warning(
            "Some model assets are missing, so predictions may fail until they are recreated: "
            + ", ".join(missing_assets)
        )


def main():
    inject_styles()
    render_hero()
    render_overview_cards()
    st.write("")
    render_missing_assets_warning()

    selected_model = render_sidebar()

    if selected_model == "Freight Cost Prediction":
        render_freight_view()
    else:
        render_invoice_flag_view()


if __name__ == "__main__":
    main()
