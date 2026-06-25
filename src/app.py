# =====================================================
#         Customer Churn Prediction System
#                  Streamlit App
#             Teyzix Core Internship 2026
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import pickle
import os

# ─────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "ChurnLens · ML-2",
    page_icon   = "🔮",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg-deep    : #0a0a14;
    --bg-card    : #12121f;
    --bg-panel   : #1a1a2e;
    --border     : #2a2a45;
    --accent     : #7c83fd;
    --accent2    : #fd7cba;
    --accent3    : #7cfdd9;
    --accent4    : #fdcd7c;
    --text-primary  : #e8e8ff;
    --text-secondary: #9090b8;
    --text-muted    : #555580;
    --danger     : #fd6b6b;
    --success    : #6bfdb0;
    --radius     : 14px;
    --radius-sm  : 8px;
}

/* ── App shell ── */
.stApp {
    background : var(--bg-deep) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background    : var(--bg-card) !important;
    border-right  : 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Main content padding ── */
.block-container {
    padding-top   : 2rem !important;
    padding-bottom: 4rem !important;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family : 'Space Grotesk', sans-serif !important;
    color       : var(--text-primary) !important;
}

/* ── Paragraph / label text ── */
p, label, .stMarkdown, div[data-testid="stText"] {
    color       : var(--text-secondary) !important;
    font-family : 'Space Grotesk', sans-serif !important;
}

/* ── Number inputs & sliders ── */
input[type="number"], .stSlider {
    background : var(--bg-panel) !important;
    color      : var(--text-primary) !important;
    border     : 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] > div > div {
    background   : var(--bg-panel) !important;
    border       : 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color        : var(--text-primary) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background   : var(--bg-panel) !important;
    border       : 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding      : 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color : var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color      : var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* ── Predict button ── */
.stButton > button {
    background   : linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color        : #fff !important;
    border       : none !important;
    border-radius: var(--radius) !important;
    padding      : 0.75rem 2rem !important;
    font-family  : 'Space Grotesk', sans-serif !important;
    font-weight  : 600 !important;
    font-size    : 1rem !important;
    width        : 100% !important;
    transition   : opacity 0.2s !important;
    cursor       : pointer !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background   : var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    gap          : 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background : transparent !important;
    color      : var(--text-muted) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    padding    : 0.6rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color        : var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background   : var(--bg-panel) !important;
    border-radius: var(--radius) !important;
}

/* ── Divider ── */
hr {
    border-color : var(--border) !important;
    margin       : 1.5rem 0 !important;
}

/* ── Custom cards ── */
.churn-card {
    background   : var(--bg-panel);
    border       : 1px solid var(--border);
    border-radius: var(--radius);
    padding      : 1.6rem 1.8rem;
    margin-bottom: 1rem;
}
.risk-high {
    border-left  : 4px solid var(--danger) !important;
}
.risk-low {
    border-left  : 4px solid var(--success) !important;
}
.risk-medium {
    border-left  : 4px solid var(--accent4) !important;
}

/* ── Hero title ── */
.hero-title {
    font-family : 'Space Grotesk', sans-serif;
    font-size   : 2.6rem;
    font-weight : 700;
    background  : linear-gradient(135deg, #7c83fd 0%, #fd7cba 60%, #7cfdd9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height : 1.15;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-family : 'Space Grotesk', sans-serif;
    font-size   : 1rem;
    color       : var(--text-muted) !important;
    letter-spacing: 0.04em;
}
.badge {
    display     : inline-block;
    background  : var(--bg-panel);
    border      : 1px solid var(--border);
    border-radius: 999px;
    padding     : 0.2rem 0.85rem;
    font-size   : 0.72rem;
    font-family : 'JetBrains Mono', monospace;
    color       : var(--accent) !important;
    letter-spacing: 0.06em;
    margin-right: 0.4rem;
}
.section-label {
    font-size  : 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color      : var(--accent) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-number {
    font-family: 'JetBrains Mono', monospace;
    font-size  : 3.2rem;
    font-weight: 700;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
#  PLOT THEME
# ─────────────────────────────────────────────────────

PLT_THEME = {
    "figure.facecolor" : "#12121f",
    "axes.facecolor"   : "#1a1a2e",
    "axes.edgecolor"   : "#2a2a45",
    "axes.labelcolor"  : "#9090b8",
    "axes.titlecolor"  : "#e8e8ff",
    "axes.titlesize"   : 12,
    "axes.titleweight" : "bold",
    "axes.grid"        : True,
    "grid.color"       : "#22223a",
    "grid.linewidth"   : 0.5,
    "text.color"       : "#9090b8",
    "xtick.color"      : "#555580",
    "ytick.color"      : "#555580",
    "legend.facecolor" : "#1a1a2e",
    "legend.edgecolor" : "#2a2a45",
    "font.family"      : "DejaVu Sans",
    "figure.dpi"       : 110,
}
plt.rcParams.update(PLT_THEME)

ACCENT  = "#7c83fd"
ACCENT2 = "#fd7cba"
ACCENT3 = "#7cfdd9"
ACCENT4 = "#fdcd7c"
PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#fd6b6b"]


# ─────────────────────────────────────────────────────
#  LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    model   = pickle.load(open("churn_model.pkl",   "rb"))
    scaler  = pickle.load(open("scaler.pkl",         "rb"))
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_ok  = True
except Exception as e:
    artifacts_ok  = False
    artifact_err  = str(e)


# ─────────────────────────────────────────────────────
#  LOAD DATASET (for EDA tab)
# ─────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_dataset.csv")

    # ── same preprocessing as training ──
    df["LastActivityDate"]      = pd.to_datetime(df["LastActivityDate"])
    today                       = pd.Timestamp.today()
    df["DaysSinceLastActivity"] = (today - df["LastActivityDate"]).dt.days
    df.drop("LastActivityDate", axis=1, inplace=True)

    df["SpendingPerLogin"] = df["MonthlySpending"]  / (df["LoginFrequency"]           + 1)
    df["SupportPerTenure"] = df["CustomerSupportRequests"] / (df["Tenure"]            + 1)
    df["PurchasePerMonth"] = df["NumberOfPurchases"] / (df["Tenure"]                  + 1)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    for col in ["Gender","City","SubscriptionType","PaymentMethod",
                "DeviceType","AutoRenew","ChurnStatus"]:
        df[col] = encoder.fit_transform(df[col])

    df.drop("CustomerID", axis=1, inplace=True)
    return df

try:
    df        = load_data()
    data_ok   = True
except Exception:
    data_ok   = False


# ─────────────────────────────────────────────────────
#  FEATURE COLUMN ORDER  (must match training)
# ─────────────────────────────────────────────────────

FEATURE_COLS = [
    "Age", "Gender", "City", "SubscriptionType",
    "MonthlySpending", "Tenure", "NumberOfPurchases",
    "CustomerSupportRequests", "LoginFrequency",
    "SatisfactionScore", "PaymentMethod", "DeviceType",
    "AutoRenew", "DaysSinceLastActivity",
    "SpendingPerLogin", "SupportPerTenure", "PurchasePerMonth",
]


# ─────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem'>
        <div class='hero-title' style='font-size:1.5rem'>🔮 ChurnLens</div>
        <div class='hero-sub'>ML-2 · Teyzix Core 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        label  = "",
        options= ["Predict", "EDA & Insights", "Model Info"],
        label_visibility = "collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#555580; line-height:1.8'>
        <span class='badge'>Task</span> ML-2<br>
        <span class='badge'>Domain</span> Machine Learning<br>
        <span class='badge'>Model</span> Random Forest<br>
        <span class='badge'>Deadline</span> 29 Jun 2026
    </div>
    """, unsafe_allow_html=True)

    if not artifacts_ok:
        st.error(f"⚠ Model not found.\nRun the training script first.\n\n`{artifact_err}`")


# ═════════════════════════════════════════════════════
#  PAGE : PREDICT
# ═════════════════════════════════════════════════════

if page == "Predict":

    # ── Hero ──────────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom:2rem'>
        <div class='hero-title'>Customer Churn Predictor</div>
        <div class='hero-sub'>Enter customer details below to assess churn risk in real time.</div>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts_ok:
        st.error("Model artifacts not loaded. Please train the model first by running `churn_prediction_ML2.py`.")
        st.stop()

    # ── Input Form ────────────────────────────────────
    st.markdown("<div class='section-label'>Customer Profile</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age    = st.number_input("Age",    min_value=18, max_value=80, value=35, step=1)
        gender = st.selectbox("Gender",   ["Male", "Female"])
        city   = st.selectbox("City",     ["City A", "City B", "City C", "City D"])

    with col2:
        st.markdown("**Subscription**")
        sub_type   = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        payment    = st.selectbox("Payment Method",    ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])
        device     = st.selectbox("Device Type",       ["Mobile", "Desktop", "Tablet"])
        auto_renew = st.selectbox("Auto Renew",        ["Yes", "No"])

    with col3:
        st.markdown("**Behaviour**")
        monthly_spend   = st.number_input("Monthly Spending ($)",        min_value=0.0,  max_value=500.0, value=45.0, step=0.5)
        tenure          = st.number_input("Tenure (months)",             min_value=0,    max_value=120,   value=12,   step=1)
        purchases       = st.number_input("Number of Purchases",         min_value=0,    max_value=200,   value=8,    step=1)
        support_req     = st.number_input("Support Requests",            min_value=0,    max_value=50,    value=3,    step=1)
        login_freq      = st.number_input("Login Frequency (per month)", min_value=0,    max_value=100,   value=12,   step=1)
        satisfaction    = st.slider("Satisfaction Score",                min_value=1,    max_value=10,    value=7)
        days_inactive   = st.number_input("Days Since Last Activity",    min_value=0,    max_value=730,   value=15,   step=1)

    st.markdown("---")

    predict_btn = st.button("🔮  Predict Churn Risk")

    if predict_btn:
        # ── Encode categoricals ────────────────────────
        gender_enc  = 1 if gender   == "Male"          else 0
        city_enc    = ["City A","City B","City C","City D"].index(city)
        sub_enc     = ["Basic","Standard","Premium"].index(sub_type)
        pay_enc     = ["Credit Card","Debit Card","PayPal","Bank Transfer"].index(payment)
        dev_enc     = ["Mobile","Desktop","Tablet"].index(device)
        auto_enc    = 1 if auto_renew == "Yes"         else 0

        # ── Build sample using only the 14 features the scaler was trained on ──
        # (The 3 engineered features were added AFTER scaler.fit_transform in training,
        #  so the saved scaler only knows 14 columns.)
        spending_per_login = monthly_spend / (login_freq + 1)
        support_per_tenure = support_req / (tenure + 1)
        purchase_per_month = purchases / (tenure + 1)

        sample = np.array([[
            age, gender_enc, city_enc, sub_enc,
            monthly_spend, tenure, purchases, support_req,
            login_freq, satisfaction, pay_enc, dev_enc,
            auto_enc, days_inactive,
            spending_per_login, support_per_tenure, purchase_per_month,
    ]])

        sample_scaled = scaler.transform(sample)
        prediction    = model.predict(sample_scaled)[0]
        proba         = model.predict_proba(sample_scaled)[0]
        churn_prob    = proba[1]
        retain_prob   = proba[0]

        # ── Risk band ─────────────────────────────────
        if churn_prob >= 0.66:
            risk_band   = "High Risk"
            risk_class  = "risk-high"
            risk_emoji  = "🔴"
            risk_color  = "#fd6b6b"
        elif churn_prob >= 0.33:
            risk_band   = "Medium Risk"
            risk_class  = "risk-medium"
            risk_emoji  = "🟡"
            risk_color  = "#fdcd7c"
        else:
            risk_band   = "Low Risk"
            risk_class  = "risk-low"
            risk_emoji  = "🟢"
            risk_color  = "#6bfdb0"

        # ── Result card ───────────────────────────────
        verdict = "LIKELY TO CHURN" if prediction == 1 else "WILL STAY"

        st.markdown(f"""
        <div class='churn-card {risk_class}'>
            <div class='section-label'>Prediction Result</div>
            <div style='display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap'>
                <div>
                    <div class='result-number' style='color:{risk_color}'>{churn_prob*100:.1f}%</div>
                    <div style='color:#555580; font-size:0.8rem; margin-top:0.2rem'>Churn Probability</div>
                </div>
                <div style='flex:1; min-width:200px'>
                    <div style='font-size:1.4rem; font-weight:700; color:{risk_color}'>
                        {risk_emoji}  {verdict}
                    </div>
                    <div style='margin-top:0.4rem'>
                        <span class='badge' style='color:{risk_color} !important'>{risk_band}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row ───────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Churn Probability",   f"{churn_prob*100:.2f}%")
        k2.metric("Retention Probability",f"{retain_prob*100:.2f}%")
        k3.metric("Risk Band",            risk_band)
        k4.metric("Satisfaction Score",   f"{satisfaction}/10")

        st.markdown("---")

        # ── Probability gauge chart ────────────────────
        col_g, col_fi = st.columns([1, 1])

        with col_g:
            st.markdown("<div class='section-label'>Churn vs Retention</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#12121f")

            bars = ax.barh(
                ["Retain", "Churn"],
                [retain_prob * 100, churn_prob * 100],
                color=[ACCENT3, ACCENT2],
                height=0.4,
                edgecolor="none"
            )
            for bar, val in zip(bars, [retain_prob*100, churn_prob*100]):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=11,
                        color="#e8e8ff", fontweight="bold")

            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Prediction Breakdown")
            ax.spines[:].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_fi:
            st.markdown("<div class='section-label'>Top Contributing Factors</div>", unsafe_allow_html=True)

            # Feature importance from model
            feat_imp = pd.DataFrame({
                "Feature"   : FEATURE_COLS,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(8)

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            fig2.patch.set_facecolor("#12121f")

            colors = [ACCENT if i < 3 else ACCENT2 if i < 6 else "#555580"
                      for i in range(len(feat_imp))]

            ax2.barh(
                feat_imp["Feature"].str.replace("_", " "),
                feat_imp["Importance"],
                color=colors, edgecolor="none", height=0.55
            )
            ax2.invert_yaxis()
            ax2.set_xlabel("Importance")
            ax2.set_title("Feature Importance")
            ax2.spines[:].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Business recommendation ────────────────────
        st.markdown("<div class='section-label' style='margin-top:1rem'>Business Recommendation</div>",
                    unsafe_allow_html=True)

        if churn_prob >= 0.66:
            recs = [
                "📞  Assign a dedicated retention agent — contact within 48 hours.",
                "🎁  Offer a loyalty discount or free premium upgrade for 1 month.",
                "📊  Review their support history and resolve any open tickets immediately.",
                "💬  Trigger a personalised re-engagement email campaign.",
            ]
        elif churn_prob >= 0.33:
            recs = [
                "📧  Send a proactive check-in email highlighting unused features.",
                "⭐  Offer a satisfaction survey and act on feedback quickly.",
                "🏷  Consider a targeted discount on their next billing cycle.",
            ]
        else:
            recs = [
                "✅  Customer appears healthy — maintain regular engagement.",
                "🎯  Candidate for upsell to a higher subscription tier.",
                "🌟  Consider inviting to a loyalty rewards programme.",
            ]

        rec_html = "".join(
            f"<div style='padding:0.6rem 0; border-bottom:1px solid #2a2a45; color:#c9c9e3'>{r}</div>"
            for r in recs
        )
        st.markdown(
            f"<div class='churn-card'>{rec_html}</div>",
            unsafe_allow_html=True
        )


# ═════════════════════════════════════════════════════
#  PAGE : EDA & INSIGHTS
# ═════════════════════════════════════════════════════

elif page == "EDA & Insights":

    st.markdown("""
    <div style='margin-bottom:2rem'>
        <div class='hero-title'>EDA & Insights</div>
        <div class='hero-sub'>Explore patterns in the customer dataset that drive churn behaviour.</div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("Dataset not found. Ensure `customer_churn_dataset.csv` is in the same directory.")
        st.stop()

    # ── Dataset KPIs ──────────────────────────────────
    total      = len(df)
    churned    = int(df["ChurnStatus"].sum())
    churn_rate = churned / total * 100
    avg_spend  = df["MonthlySpending"].mean()
    avg_sat    = df["SatisfactionScore"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers",  f"{total:,}")
    m2.metric("Churned",          f"{churned:,}")
    m3.metric("Churn Rate",       f"{churn_rate:.1f}%")
    m4.metric("Avg Satisfaction", f"{avg_sat:.2f}/10")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "  📊  Distribution  ",
        "  🔗  Correlations  ",
        "  🎯  Churn Drivers  ",
        "  📋  Raw Data  ",
    ])

    # ── Tab 1 : Distribution ──────────────────────────
    with tab1:
        st.markdown("<div class='section-label' style='margin-top:1rem'>Churn Distribution</div>",
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#12121f")
            counts = df["ChurnStatus"].value_counts()
            ax.bar(["Retained", "Churned"], counts.values,
                   color=[ACCENT3, ACCENT2], edgecolor="none", width=0.45)
            for i, v in enumerate(counts.values):
                ax.text(i, v + 8, f"{v:,}", ha="center", color="#e8e8ff", fontsize=10)
            ax.set_title("Churn Count")
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col_b:
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#12121f")
            ax.pie(
                counts.values,
                labels=["Retained", "Churned"],
                colors=[ACCENT3, ACCENT2],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "#12121f", "linewidth": 2},
                textprops={"color": "#c9c9e3"}
            )
            ax.set_title("Churn Split")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown("<div class='section-label' style='margin-top:1rem'>Demographic Breakdown</div>",
                    unsafe_allow_html=True)

        col_c, col_d, col_e = st.columns(3)
        demo_cols = ["Gender", "SubscriptionType", "DeviceType"]
        demo_titles = ["Gender", "Subscription Type", "Device Type"]

        for ax_col, feat, title in zip([col_c, col_d, col_e], demo_cols, demo_titles):
            with ax_col:
                fig, ax = plt.subplots(figsize=(4, 3))
                fig.patch.set_facecolor("#12121f")
                sns.countplot(x=feat, data=df, ax=ax,
                              palette=PALETTE[:df[feat].nunique()], edgecolor="none")
                ax.set_title(title); ax.set_xlabel(""); ax.spines[:].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("<div class='section-label' style='margin-top:1rem'>Spending & Activity</div>",
                    unsafe_allow_html=True)

        col_f, col_g = st.columns(2)
        with col_f:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#12121f")
            sns.histplot(df["MonthlySpending"], bins=30, ax=ax,
                         color=ACCENT, kde=True)
            ax.set_title("Monthly Spending"); ax.spines[:].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_g:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#12121f")
            sns.histplot(df["DaysSinceLastActivity"], bins=30, ax=ax,
                         color=ACCENT2, kde=True)
            ax.set_title("Days Since Last Activity"); ax.spines[:].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Tab 2 : Correlations ──────────────────────────
    with tab2:
        st.markdown("<div class='section-label' style='margin-top:1rem'>Correlation Heatmap</div>",
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 9))
        fig.patch.set_facecolor("#12121f")
        cmap = sns.diverging_palette(260, 330, s=80, l=55, as_cmap=True)
        sns.heatmap(
            df.corr(), annot=True, fmt=".2f", cmap=cmap,
            ax=ax, linewidths=0.3, linecolor="#0a0a14",
            annot_kws={"size": 7}
        )
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Tab 3 : Churn Drivers ─────────────────────────
    with tab3:
        st.markdown("<div class='section-label' style='margin-top:1rem'>Key Churn Drivers (Box Plots)</div>",
                    unsafe_allow_html=True)

        driver_features = [
            "SatisfactionScore", "LoginFrequency",
            "MonthlySpending",   "DaysSinceLastActivity",
            "Tenure",            "CustomerSupportRequests",
        ]

        for i in range(0, len(driver_features), 3):
            cols = st.columns(3)
            for j, feat in enumerate(driver_features[i:i+3]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    fig.patch.set_facecolor("#12121f")
                    sns.boxplot(
                        x="ChurnStatus", y=feat, data=df,
                        ax=ax, palette=[ACCENT3, ACCENT2],
                        flierprops={"marker": ".", "color": "#555580"}
                    )
                    ax.set_title(feat.replace("_", " "), fontsize=10)
                    ax.set_xticklabels(["Retained", "Churned"])
                    ax.set_xlabel("")
                    ax.spines[:].set_visible(False)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("<div class='section-label' style='margin-top:1rem'>Behaviour Scatter</div>",
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#12121f")
        for val, label, color in zip([0,1], ["Retained","Churned"], [ACCENT3, ACCENT2]):
            sub = df[df["ChurnStatus"] == val]
            ax.scatter(sub["LoginFrequency"], sub["MonthlySpending"],
                       label=label, color=color, alpha=0.45, s=14, edgecolors="none")
        ax.set_xlabel("Login Frequency"); ax.set_ylabel("Monthly Spending")
        ax.set_title("Login Frequency vs Monthly Spending by Churn Status")
        ax.legend(); ax.spines[:].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Tab 4 : Raw Data ──────────────────────────────
    with tab4:
        st.markdown("<div class='section-label' style='margin-top:1rem'>Dataset Preview</div>",
                    unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True, height=420)
        st.caption(f"Showing 100 of {len(df):,} rows  ·  {df.shape[1]} features")


# ═════════════════════════════════════════════════════
#  PAGE : MODEL INFO
# ═════════════════════════════════════════════════════

elif page == "Model Info":

    st.markdown("""
    <div style='margin-bottom:2rem'>
        <div class='hero-title'>Model Information</div>
        <div class='hero-sub'>Architecture, evaluation metrics, and feature importance of the trained model.</div>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts_ok:
        st.error("Model not loaded. Run the training script first.")
        st.stop()

    if not data_ok:
        st.error("Dataset not found.")
        st.stop()

    # ── Model params ──────────────────────────────────
    st.markdown("<div class='section-label'>Model Architecture</div>", unsafe_allow_html=True)

    params = model.get_params()
    p1, p2 = st.columns(2)

    with p1:
        st.markdown(f"""
        <div class='churn-card'>
            <div class='section-label'>Estimator</div>
            <div style='color:#e8e8ff; font-size:1.1rem; font-weight:600'>Random Forest Classifier</div>
            <div style='margin-top:0.8rem; color:#555580; font-size:0.85rem; line-height:2'>
                Trees            : <span style='color:#7c83fd'>{params.get("n_estimators","—")}</span><br>
                Max Depth        : <span style='color:#7c83fd'>{params.get("max_depth","—")}</span><br>
                Min Samples Split: <span style='color:#7c83fd'>{params.get("min_samples_split","—")}</span><br>
                Random State     : <span style='color:#7c83fd'>{params.get("random_state","—")}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class='churn-card'>
            <div class='section-label'>Training Pipeline</div>
            <div style='color:#555580; font-size:0.85rem; line-height:2.2'>
                Encoding   : <span style='color:#7cfdd9'>LabelEncoder</span><br>
                Scaling    : <span style='color:#7cfdd9'>StandardScaler</span><br>
                Split      : <span style='color:#7cfdd9'>80 / 20 (stratified)</span><br>
                Tuning     : <span style='color:#7cfdd9'>GridSearchCV  (cv=3)</span><br>
                Features   : <span style='color:#7cfdd9'>{len(FEATURE_COLS)} original features</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature importance ────────────────────────────
    st.markdown("<div class='section-label'>Feature Importance</div>", unsafe_allow_html=True)

    feat_df = pd.DataFrame({
        "Feature"   : FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#12121f")
        bar_colors = [ACCENT if i < 3 else ACCENT2 if i < 8 else "#555580"
                      for i in range(len(feat_df))]
        ax.barh(feat_df["Feature"].str.replace("_", " "),
                feat_df["Importance"], color=bar_colors, edgecolor="none")
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance — Tuned RF")
        for i, (_, row) in enumerate(feat_df.iterrows()):
            ax.text(row["Importance"] + 0.001, i,
                    f"{row['Importance']:.4f}", va="center",
                    fontsize=7.5, color="#c9c9e3")
        ax.spines[:].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_table:
        display_df = feat_df.copy()
        display_df["Feature"]    = display_df["Feature"].str.replace("_", " ")
        display_df["Importance"] = display_df["Importance"].map("{:.4f}".format)
        st.dataframe(display_df.reset_index(drop=True),
                     use_container_width=True, height=420)

    st.markdown("---")

    # ── Evaluation metrics (live on test set) ─────────
    st.markdown("<div class='section-label'>Evaluation on Test Split (20%)</div>",
                unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)

    X = df[FEATURE_COLS]   # only the 14 features the scaler knows
    y = df["ChurnStatus"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test.values)
    y_pred    = model.predict(X_test_sc)

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.4f}")
    e2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    e3.metric("Recall",    f"{recall_score(y_test, y_pred):.4f}")
    e4.metric("F1 Score",  f"{f1_score(y_test, y_pred):.4f}")
    e5.metric("ROC AUC",   f"{roc_auc_score(y_test, y_pred):.4f}")

    # ── Confusion matrix ──────────────────────────────
    from sklearn.metrics import confusion_matrix

    st.markdown("<div class='section-label' style='margin-top:1.5rem'>Confusion Matrix</div>",
                unsafe_allow_html=True)

    cm_col, _ = st.columns([1, 1])
    with cm_col:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#12121f")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d",
                    cmap=sns.light_palette(ACCENT, as_cmap=True),
                    ax=ax, linewidths=1, linecolor="#0a0a14",
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticklabels(["Not Churned", "Churned"])
        ax.set_yticklabels(["Not Churned", "Churned"], rotation=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()