
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "car price prediction model" / "Car details v3.csv"
MODEL_PATH = BASE_DIR / "autoiq_random_forest.pkl"

FEATURES = [
    "year",
    "km_driven",
    "mileage_num",
    "engine_num",
    "max_power_num",
    "torque_num",
    "torque_rpm_min",
    "torque_rpm_max",
    "seats",
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

st.set_page_config(
    page_title="AutoIQ | Intelligent Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------------- GENERAL ---------------- */

.stApp {
    background-color: #0b1020;
    color: #f4f7ff;
    font-family: 'Inter', sans-serif;
}

/* Increase general text size */

html, body, [data-testid="stMarkdownContainer"] {
    font-size: 16px;
}

p, li {
    font-size: 16px;
    line-height: 1.65;
}

label {
    font-size: 15px !important;
    font-weight: 600 !important;
}

[data-testid="stCaptionContainer"] {
    font-size: 14px !important;
}

/* ---------------- HEADINGS ---------------- */

h1 {
    font-size: 36px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}

h2 {
    font-size: 27px !important;
    font-weight: 750 !important;
    color: #ffffff !important;
}

h3 {
    font-size: 21px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* ---------------- SIDEBAR ---------------- */

[data-testid="stSidebar"] {
    background: #10182b;
    border-right: 1px solid #263752;
}

[data-testid="stSidebar"] p {
    font-size: 15px;
}

[data-testid="stSidebar"] label {
    font-size: 16px !important;
}

/* ---------------- HERO ---------------- */

.hero {
    padding: 32px;
    border-radius: 20px;

    background:
        linear-gradient(
            120deg,
            rgba(30, 64, 115, 0.95),
            rgba(21, 37, 67, 0.95)
        );

    border: 1px solid rgba(130, 180, 255, 0.25);

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.25),
        inset 0 1px 1px rgba(255, 255, 255, 0.08);

    margin-bottom: 28px;
}

.hero h1 {
    font-size: 40px !important;
    margin-bottom: 10px;
}

.hero p {
    font-size: 17px;
    color: #c1d2ec;
}

/* ---------------- METRIC CARDS ---------------- */

.metric-card {
    background: linear-gradient(
        145deg,
        #17243c,
        #111b2e
    );

    border: 1px solid rgba(150, 190, 255, 0.16);
    border-radius: 16px;

    padding: 22px;
    min-height: 130px;

    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);

    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(100, 180, 255, 0.5);
}

.metric-label {
    color: #aabbd6;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.6px;
}

.metric-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #8fa2c0;
    font-size: 13px;
    margin-top: 5px;
}

/* ---------------- VALUATION CARD ---------------- */

.price-card {
    background: linear-gradient(
        125deg,
        #153b4a,
        #153d3d
    );

    border: 1px solid rgba(60, 220, 180, 0.4);
    padding: 30px;
    border-radius: 20px;
    margin: 20px 0;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.2),
        inset 0 1px 2px rgba(255, 255, 255, 0.1);
}

.price-label {
    color: #b7e9dd;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 1px;
}

.price-value {
    color: #ffffff;
    font-size: 46px;
    font-weight: 850;
    margin-top: 10px;
}

.small-muted {
    color: #a9bdc8;
    font-size: 14px;
}

/* ---------------- GLASS BUTTONS ---------------- */

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {

    background: linear-gradient(
        135deg,
        rgba(100, 160, 255, 0.22),
        rgba(100, 220, 210, 0.10)
    ) !important;

    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);

    border: 1px solid rgba(180, 215, 255, 0.35) !important;

    border-radius: 13px !important;

    color: #ffffff !important;

    font-size: 16px !important;
    font-weight: 700 !important;

    min-height: 48px;

    padding: 10px 20px !important;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.22),
        inset 0 1px 2px rgba(255, 255, 255, 0.15);

    transition: all 0.25s ease-in-out;
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {

    background: linear-gradient(
        135deg,
        rgba(100, 160, 255, 0.42),
        rgba(100, 220, 210, 0.24)
    ) !important;

    border-color: rgba(180, 225, 255, 0.75) !important;

    box-shadow:
        0 0 22px rgba(80, 170, 255, 0.25),
        inset 0 1px 3px rgba(255, 255, 255, 0.2);

    transform: translateY(-2px);
}

[data-testid="stButton"] button:active {
    transform: scale(0.98);
}

/* ---------------- INPUT FIELDS ---------------- */

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    font-size: 16px !important;
}

[data-testid="stSelectbox"] div {
    font-size: 15px;
}

/* ---------------- DATAFRAMES ---------------- */

[data-testid="stDataFrame"] {
    border: 1px solid #263752;
    border-radius: 12px;
    overflow: hidden;
}

/* ---------------- DIVIDERS ---------------- */

hr {
    border-color: rgba(150, 180, 230, 0.16);
}

/* ---------------- RESPONSIVE ---------------- */

@media (max-width: 768px) {
    .hero h1 {
        font-size: 29px !important;
    }

    .price-value {
        font-size: 34px;
    }

    h1 {
        font-size: 30px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def extract_number(value):

    if pd.isna(value):
        return np.nan

    match = re.search(r"\d+\.?\d*", str(value))

    if match:
        return float(match.group())

    return np.nan


def extract_torque(value):

    if pd.isna(value):
        return np.nan

    s = str(value).lower().strip()

    match = re.search(r"^\s*(\d+(?:\.\d+)?)", s)

    if not match:
        return np.nan

    torque = float(match.group(1))

    if "nm" in s:
        return torque

    if "kgm" in s and torque <= 50:
        return torque * 9.80665

    return torque


def extract_rpm(value):

    if pd.isna(value):
        return np.nan, np.nan

    s = str(value).lower().replace(",", "")

    match = re.search(
        r"(?:@|\bat\b|/)\s*(\d+)"
        r"(?:\s*-\s*(\d+))?"
        r"(?:\s*\+/-\s*(\d+))?",
        s
    )

    if not match:
        return np.nan, np.nan

    rpm_min = float(match.group(1))

    if match.group(3):

        tolerance = float(match.group(3))

        rpm_max = rpm_min + tolerance
        rpm_min = rpm_min - tolerance

    elif match.group(2):

        rpm_max = float(match.group(2))

    else:

        rpm_max = rpm_min

    if rpm_min < 500 or rpm_max > 6000:
        return np.nan, np.nan

    return rpm_min, rpm_max


def prepare_dataset(df):

    df = df.copy().drop_duplicates()

    df["brand"] = df["name"].astype("string").str.split().str[0]

    df["mileage_num"] = df["mileage"].apply(extract_number)
    df["engine_num"] = df["engine"].apply(extract_number)
    df["max_power_num"] = df["max_power"].apply(extract_number)

    df["torque_num"] = df["torque"].apply(extract_torque)

    df[["torque_rpm_min", "torque_rpm_max"]] = pd.DataFrame(
        df["torque"].apply(extract_rpm).tolist(),
        index=df.index
    )

    df.loc[df["mileage_num"] <= 0, "mileage_num"] = np.nan
    df.loc[df["engine_num"] <= 0, "engine_num"] = np.nan
    df.loc[df["max_power_num"] <= 0, "max_power_num"] = np.nan
    df.loc[df["torque_num"] <= 0, "torque_num"] = np.nan
    df.loc[df["torque_num"] > 700, "torque_num"] = np.nan

    numeric_columns = [
        "year",
        "selling_price",
        "km_driven",
        "mileage_num",
        "engine_num",
        "max_power_num",
        "torque_num",
        "torque_rpm_min",
        "torque_rpm_max",
        "seats",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["selling_price", "year", "km_driven"])

    df = df[
        (df["selling_price"] > 0)
        & (df["km_driven"] >= 0)
    ]

    return df


# ============================================================
# 4. LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_dataset():

    raw = pd.read_csv(DATA_PATH)

    return prepare_dataset(raw)


@st.cache_resource
def load_model():

    loaded_model = joblib.load(MODEL_PATH)

    if list(loaded_model.feature_names_in_) != FEATURES:
        raise ValueError("Model feature configuration mismatch.")

    return loaded_model


try:

    df = load_dataset()
    model = load_model()

except Exception as error:

    st.error(f"Unable to load AutoIQ: {error}")

    st.info(
        "Check your dataset and model paths, then restart Streamlit."
    )

    st.stop()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def format_inr(value):

    number = int(round(value))

    sign = "-" if number < 0 else ""
    number = abs(number)

    last_three = str(number % 1000)
    remaining = number // 1000

    groups = []

    while remaining > 0:

        groups.append(str(remaining % 100))
        remaining //= 100

    if groups:

        formatted = ",".join(reversed(groups)) + "," + last_three

    else:

        formatted = last_three

    return "₹" + sign + formatted


def median_or_default(series, fallback):

    value = series.median()

    if pd.isna(value):
        return fallback

    return float(value)


def metric_card(label, value, note=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def predict_price(input_data):

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    prediction = model.predict(input_df)[0]

    return max(0, float(prediction))


def save_prediction(input_data, prediction):

    if "history" not in st.session_state:
        st.session_state.history = []

    record = input_data.copy()

    record["predicted_price"] = prediction
    record["timestamp"] = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.session_state.history.insert(0, record)


def get_comparables(car):

    comparable = df.copy()

    comparable = comparable[
        comparable["brand"] == car["brand"]
    ]

    comparable = comparable[
        comparable["fuel"] == car["fuel"]
    ]

    comparable = comparable[
        (comparable["year"] >= car["year"] - 3)
        & (comparable["year"] <= car["year"] + 3)
    ]

    return comparable


# ============================================================
# 6. VEHICLE INPUT FORM
# ============================================================

def car_input_form(prefix):

    st.markdown("#### Vehicle specifications")

    c1, c2 = st.columns(2)

    brands = sorted(df["brand"].dropna().unique().tolist())
    fuels = sorted(df["fuel"].dropna().unique().tolist())
    sellers = sorted(df["seller_type"].dropna().unique().tolist())
    transmissions = sorted(df["transmission"].dropna().unique().tolist())
    owners = sorted(df["owner"].dropna().unique().tolist())

    defaults = {
        "year": int(df["year"].median()),
        "km": int(df["km_driven"].median()),
        "mileage": median_or_default(df["mileage_num"], 18),
        "engine": median_or_default(df["engine_num"], 1200),
        "power": median_or_default(df["max_power_num"], 80),
        "torque": median_or_default(df["torque_num"], 150),
        "rpm_min": median_or_default(df["torque_rpm_min"], 2000),
        "rpm_max": median_or_default(df["torque_rpm_max"], 2500),
    }

    with c1:

        brand = st.selectbox(
            "Car brand",
            brands,
            key=f"{prefix}_brand"
        )

        year = st.number_input(
            "Manufacturing year",
            min_value=1990,
            max_value=2026,
            value=max(1990, min(2026, defaults["year"])),
            key=f"{prefix}_year"
        )

        km = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=1000000,
            value=max(0, defaults["km"]),
            step=1000,
            key=f"{prefix}_km"
        )

        mileage = st.number_input(
            "Mileage (km/l)",
            min_value=1.0,
            max_value=100.0,
            value=max(1.0, min(100.0, defaults["mileage"])),
            step=0.1,
            key=f"{prefix}_mileage"
        )

        engine = st.number_input(
            "Engine capacity (CC)",
            min_value=300.0,
            max_value=10000.0,
            value=max(300.0, min(10000.0, defaults["engine"])),
            step=10.0,
            key=f"{prefix}_engine"
        )

    with c2:

        power = st.number_input(
            "Maximum power (bhp)",
            min_value=10.0,
            max_value=2000.0,
            value=max(10.0, min(2000.0, defaults["power"])),
            step=1.0,
            key=f"{prefix}_power"
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=10.0,
            max_value=1500.0,
            value=max(10.0, min(1500.0, defaults["torque"])),
            step=1.0,
            key=f"{prefix}_torque"
        )

        rpm_min = st.number_input(
            "Minimum torque RPM",
            min_value=500.0,
            max_value=6000.0,
            value=max(500.0, min(6000.0, defaults["rpm_min"])),
            step=100.0,
            key=f"{prefix}_rpm_min"
        )

        rpm_max = st.number_input(
            "Maximum torque RPM",
            min_value=500.0,
            max_value=6000.0,
            value=max(500.0, min(6000.0, defaults["rpm_max"])),
            step=100.0,
            key=f"{prefix}_rpm_max"
        )

        seats = st.selectbox(
            "Number of seats",
            [2, 4, 5, 6, 7, 8, 9, 10],
            index=2,
            key=f"{prefix}_seats"
        )

    a, b, c = st.columns(3)

    with a:

        fuel = st.selectbox(
            "Fuel type",
            fuels,
            key=f"{prefix}_fuel"
        )

    with b:

        transmission = st.selectbox(
            "Transmission",
            transmissions,
            key=f"{prefix}_transmission"
        )

    with c:

        seller_type = st.selectbox(
            "Seller type",
            sellers,
            key=f"{prefix}_seller"
        )

    owner = st.selectbox(
        "Ownership history",
        owners,
        key=f"{prefix}_owner"
    )

    return {
        "year": year,
        "km_driven": km,
        "mileage_num": mileage,
        "engine_num": engine,
        "max_power_num": power,
        "torque_num": torque,
        "torque_rpm_min": rpm_min,
        "torque_rpm_max": rpm_max,
        "seats": seats,
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🚘 AutoIQ")

    st.caption("INTELLIGENT USED CAR VALUATION")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "AI Valuation",
            "Compare Cars",
            "Market Insights",
            "Prediction History",
            "Dataset Explorer",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model information")

    st.caption("Algorithm: Random Forest")
    st.caption("Dataset: Used Car Details")
    st.caption(f"Vehicles: {len(df):,}")
    st.caption("Features: 14 engineered inputs")

    st.divider()

    st.caption("AutoIQ | ML-powered valuation")


# ============================================================
# 8. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Understand the value of every drive.</h1>
        <p>
            Intelligent used car valuation powered by machine learning.
            Explore vehicle prices, market trends and predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)

    with a:
        metric_card(
            "VEHICLES ANALYZED",
            f"{len(df):,}",
            "Cleaned dataset records"
        )

    with b:
        metric_card(
            "AVERAGE PRICE",
            format_inr(df["selling_price"].mean()),
            "Across available vehicles"
        )

    with c:
        metric_card(
            "MEDIAN PRICE",
            format_inr(df["selling_price"].median()),
            "Middle of price distribution"
        )

    with d:
        metric_card(
            "CAR BRANDS",
            str(df["brand"].nunique()),
            "Unique brands"
        )

    st.markdown("## Market overview")

    left, right = st.columns(2)

    # --------------------------------------------------------
    # PRICE DISTRIBUTION LINE GRAPH
    # --------------------------------------------------------

    with left:

        st.markdown("### Price distribution")

        price_limit = df["selling_price"].quantile(0.99)

        prices = df.loc[
            df["selling_price"] <= price_limit,
            "selling_price"
        ]

        counts, edges = np.histogram(prices, bins=45)

        midpoints = (edges[:-1] + edges[1:]) / 2

        price_data = pd.DataFrame({
            "Price (₹ Lakhs)": midpoints / 100000,
            "Number of Cars": counts
        })

        fig = px.line(
            price_data,
            x="Price (₹ Lakhs)",
            y="Number of Cars",
            markers=True
        )

        fig.update_traces(
            line=dict(
                color="#62C5FF",
                width=4,
                shape="spline"
            ),
            marker=dict(
                size=5,
                color="#A8E3FF"
            ),
            fill="tozeroy",
            fillcolor="rgba(98,197,255,0.12)",
            hovertemplate=(
                "Price: ₹%{x:.2f} Lakhs<br>"
                "Cars: %{y}<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            title="Used car price distribution",
            title_font_size=19,
            title_font_color="#FFFFFF",
            xaxis_title="Selling price (₹ Lakhs)",
            yaxis_title="Number of cars",
            hovermode="x unified",
            margin=dict(l=35, r=20, t=65, b=45),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.09)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Top 1% of prices excluded from this visualization "
            "to improve readability."
        )

    # --------------------------------------------------------
    # FUEL DONUT WITH LAYERED DEPTH EFFECT
    # --------------------------------------------------------

    with right:

        st.markdown("### Fuel distribution")

        fuel_counts = df["fuel"].value_counts()

        labels = fuel_counts.index.tolist()
        values = fuel_counts.values.tolist()

        colors = [
            "#6675FF",
            "#FF6347",
            "#00CFA6",
            "#B46BFF",
            "#FFB84D",
            "#36BFFA"
        ][:len(labels)]

        shadow_colors = [
            "#303A91",
            "#993C32",
            "#087A67",
            "#633C91",
            "#94652D",
            "#246B91"
        ][:len(labels)]

        fig = go.Figure()

        # Lower, darker layer creates visual depth

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.52,
            sort=False,
            direction="clockwise",
            rotation=90,
            marker=dict(
                colors=shadow_colors,
                line=dict(width=0)
            ),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
            domain=dict(
                x=[0.02, 0.76],
                y=[0.02, 0.86]
            )
        ))

        # Main donut layer

        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.52,
            sort=False,
            direction="clockwise",
            rotation=90,
            marker=dict(
                colors=colors,
                line=dict(
                    color="#171B2B",
                    width=3
                )
            ),
            textinfo="percent",
            textposition="inside",
            textfont=dict(
                size=14,
                color="white"
            ),
            showlegend=True,
            domain=dict(
                x=[0.02, 0.76],
                y=[0.10, 0.94]
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Vehicles: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            )
        ))

        fig.update_layout(
            template="plotly_dark",
            height=430,
            showlegend=True,
            legend=dict(
                orientation="v",
                x=0.79,
                y=0.5,
                xanchor="left",
                font=dict(size=13)
            ),
            margin=dict(
                l=5,
                r=5,
                t=20,
                b=10
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # BRAND ANALYSIS
    # --------------------------------------------------------

    st.markdown("### Brand-wise median price")

    brand_prices = (
        df.groupby("brand")["selling_price"]
        .median()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )

    fig = px.bar(
        brand_prices,
        x="brand",
        y="selling_price",
        color="selling_price",
        color_continuous_scale="Blues",
        labels={
            "brand": "Car brand",
            "selling_price": "Median selling price (₹)"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        coloraxis_showscale=False,
        xaxis_title="Brand",
        yaxis_title="Median price (₹)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Dashboard statistics describe the dataset, not individual model predictions."
    )


# ============================================================
# 9. AI VALUATION
# ============================================================

elif page == "AI Valuation":

    st.markdown("# AI Car Valuation")

    st.write(
        "Enter the vehicle specifications to estimate its "
        "market value using the trained Random Forest model."
    )

    with st.form("valuation_form"):

        car = car_input_form("valuation")

        submitted = st.form_submit_button(
            "🚀 Estimate Vehicle Value",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if car["torque_rpm_min"] > car["torque_rpm_max"]:

            st.error(
                "Minimum torque RPM cannot exceed maximum torque RPM."
            )

        else:

            try:

                prediction = predict_price(car)

                save_prediction(car, prediction)

                st.markdown(
                    f"""
                    <div class="price-card">
                        <div class="price-label">
                            ESTIMATED VEHICLE VALUE
                        </div>
                        <div class="price-value">
                            {format_inr(prediction)}
                        </div>
                        <div class="small-muted">
                            Predicted by AutoIQ Random Forest model
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### Vehicle summary")

                a, b, c = st.columns(3)

                a.metric("Brand", car["brand"])
                b.metric("Year", car["year"])
                c.metric(
                    "Kilometers driven",
                    f'{car["km_driven"]:,} km'
                )

                comparables = get_comparables(car)

                if not comparables.empty:

                    median_price = comparables["selling_price"].median()

                    st.markdown("### Market comparison")

                    a, b = st.columns(2)

                    a.metric(
                        "Comparable market median",
                        format_inr(median_price)
                    )

                    difference = prediction - median_price

                    percentage = (
                        difference / median_price * 100
                        if median_price > 0 else 0
                    )

                    b.metric(
                        "Difference from median",
                        format_inr(abs(difference)),
                        delta=f"{percentage:+.1f}%"
                    )

                    st.caption(
                        f"Based on {len(comparables)} comparable dataset vehicles."
                    )

                st.info(
                    "This is a machine learning estimate, not a guaranteed "
                    "selling price. Vehicle condition, location, accident "
                    "history and service records can affect actual value."
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")


# ============================================================
# 10. COMPARE CARS
# ============================================================

elif page == "Compare Cars":

    st.markdown("# Compare Vehicles")

    st.write(
        "Compare two vehicles using predictions from the same trained model."
    )

    with st.form("compare_form"):

        left, right = st.columns(2)

        with left:

            st.markdown("## 🚗 Vehicle A")

            car_a = car_input_form("compare_a")

        with right:

            st.markdown("## 🚙 Vehicle B")

            car_b = car_input_form("compare_b")

        submitted = st.form_submit_button(
            "Compare Vehicles",
            type="primary",
            use_container_width=True
        )

    if submitted:

        invalid_a = (
            car_a["torque_rpm_min"] > car_a["torque_rpm_max"]
        )

        invalid_b = (
            car_b["torque_rpm_min"] > car_b["torque_rpm_max"]
        )

        if invalid_a or invalid_b:

            st.error("Check the torque RPM ranges for both vehicles.")

        else:

            try:

                price_a = predict_price(car_a)
                price_b = predict_price(car_b)

                save_prediction(car_a, price_a)
                save_prediction(car_b, price_b)

                st.markdown("## Prediction results")

                a, b = st.columns(2)

                with a:

                    st.markdown("### Vehicle A")

                    st.markdown(
                        f"""
                        <div class="price-card">
                            <div class="price-label">
                                PREDICTED PRICE
                            </div>
                            <div class="price-value">
                                {format_inr(price_a)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with b:

                    st.markdown("### Vehicle B")

                    st.markdown(
                        f"""
                        <div class="price-card">
                            <div class="price-label">
                                PREDICTED PRICE
                            </div>
                            <div class="price-value">
                                {format_inr(price_b)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                difference = abs(price_a - price_b)

                winner = (
                    "Vehicle A"
                    if price_a > price_b
                    else "Vehicle B"
                    if price_b > price_a
                    else "Both vehicles"
                )

                st.success(
                    f"{winner} has the higher predicted value "
                    f"by {format_inr(difference)}."
                )

                comparison = pd.DataFrame({
                    "Specification": [
                        "Brand",
                        "Year",
                        "Kilometers driven",
                        "Engine (CC)",
                        "Power (bhp)",
                        "Mileage (km/l)",
                        "Fuel",
                        "Transmission",
                        "Predicted price",
                    ],

                    "Vehicle A": [
                        car_a["brand"],
                        car_a["year"],
                        car_a["km_driven"],
                        car_a["engine_num"],
                        car_a["max_power_num"],
                        car_a["mileage_num"],
                        car_a["fuel"],
                        car_a["transmission"],
                        format_inr(price_a),
                    ],

                    "Vehicle B": [
                        car_b["brand"],
                        car_b["year"],
                        car_b["km_driven"],
                        car_b["engine_num"],
                        car_b["max_power_num"],
                        car_b["mileage_num"],
                        car_b["fuel"],
                        car_b["transmission"],
                        format_inr(price_b),
                    ]
                })

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

                chart_data = pd.DataFrame({
                    "Vehicle": ["Vehicle A", "Vehicle B"],
                    "Predicted Price": [price_a, price_b]
                })

                fig = px.bar(
                    chart_data,
                    x="Vehicle",
                    y="Predicted Price",
                    text_auto=".2s",
                    color="Vehicle",
                    color_discrete_sequence=[
                        "#62C5FF",
                        "#00CFA6"
                    ]
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as error:

                st.error(f"Comparison failed: {error}")


# ============================================================
# 11. MARKET INSIGHTS
# ============================================================

elif page == "Market Insights":

    st.markdown("# Market Insights")

    st.write(
        "Explore vehicle prices, brand performance and market trends."
    )

    a, b, c = st.columns(3)

    with a:

        selected_brands = st.multiselect(
            "Car brands",
            sorted(df["brand"].dropna().unique()),
            default=[]
        )

    with b:

        selected_fuels = st.multiselect(
            "Fuel types",
            sorted(df["fuel"].dropna().unique()),
            default=[]
        )

    with c:

        selected_transmissions = st.multiselect(
            "Transmission",
            sorted(df["transmission"].dropna().unique()),
            default=[]
        )

    filtered = df.copy()

    if selected_brands:

        filtered = filtered[
            filtered["brand"].isin(selected_brands)
        ]

    if selected_fuels:

        filtered = filtered[
            filtered["fuel"].isin(selected_fuels)
        ]

    if selected_transmissions:

        filtered = filtered[
            filtered["transmission"].isin(selected_transmissions)
        ]

    if filtered.empty:

        st.warning("No vehicles match these filters.")

    else:

        a, b, c = st.columns(3)

        a.metric("Vehicles", f"{len(filtered):,}")

        b.metric(
            "Average price",
            format_inr(filtered["selling_price"].mean())
        )

        c.metric(
            "Median price",
            format_inr(filtered["selling_price"].median())
        )

        st.markdown("### Price distribution")

        fig = px.histogram(
            filtered,
            x="selling_price",
            color="fuel",
            nbins=40,
            labels={
                "selling_price": "Selling price (₹)"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Brand-wise median price")

        brand_analysis = (
            filtered.groupby("brand")["selling_price"]
            .agg(["median", "count"])
            .reset_index()
            .sort_values("median", ascending=False)
            .head(15)
        )

        fig = px.bar(
            brand_analysis,
            x="brand",
            y="median",
            color="median",
            color_continuous_scale="Blues",
            hover_data=["count"],
            labels={
                "brand": "Brand",
                "median": "Median selling price (₹)",
                "count": "Vehicle count"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # LINE GRAPH: MEDIAN PRICE BY YEAR

        st.markdown("### Manufacturing year vs median price")

        year_analysis = (
            filtered.groupby("year")["selling_price"]
            .median()
            .reset_index()
            .sort_values("year")
        )

        fig = px.line(
            year_analysis,
            x="year",
            y="selling_price",
            markers=True,
            labels={
                "year": "Manufacturing year",
                "selling_price": "Median selling price (₹)"
            }
        )

        fig.update_traces(
            line=dict(
                color="#00CFA6",
                width=4,
                shape="spline"
            ),
            marker=dict(
                size=8,
                color="#9CFFE6"
            ),
            fill="tozeroy",
            fillcolor="rgba(0,207,166,0.09)"
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.09)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 12. PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.markdown("# Prediction History")

    history = st.session_state.get("history", [])

    if not history:

        st.info(
            "No predictions yet. Visit AI Valuation or Compare Cars "
            "to create your first prediction."
        )

    else:

        history_df = pd.DataFrame(history)

        st.metric(
            "Total predictions",
            len(history_df)
        )

        display_df = history_df.copy()

        display_df["predicted_price"] = display_df[
            "predicted_price"
        ].apply(format_inr)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download prediction history",
            data=csv,
            file_name="autoiq_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear prediction history"):

            st.session_state.history = []

            st.rerun()

        st.caption(
            "History is stored in the current Streamlit session. "
            "It is cleared when the session is reset."
        )


# ============================================================
# 13. DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.markdown("# Dataset Explorer")

    st.write(
        "Explore the used car dataset powering AutoIQ."
    )

    a, b, c = st.columns(3)

    a.metric("Total records", f"{len(df):,}")

    b.metric("Unique brands", f"{df['brand'].nunique():,}")

    c.metric("Dataset columns", f"{len(df.columns):,}")

    st.markdown("### Filter dataset")

    search_brand = st.selectbox(
        "Select brand",
        ["All"] + sorted(df["brand"].dropna().unique().tolist())
    )

    explorer_df = df.copy()

    if search_brand != "All":

        explorer_df = explorer_df[
            explorer_df["brand"] == search_brand
        ]

    columns_to_show = [
        "name",
        "year",
        "selling_price",
        "km_driven",
        "fuel",
        "transmission",
        "owner",
        "mileage_num",
        "engine_num",
        "max_power_num",
        "seats",
    ]

    st.dataframe(
        explorer_df[columns_to_show],
        use_container_width=True,
        hide_index=True
    )

    csv = explorer_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download filtered dataset",
        data=csv,
        file_name="autoiq_dataset.csv",
        mime="text/csv"
    )

    st.markdown("### Dataset statistics")

    st.dataframe(
        df[
            [
                "selling_price",
                "year",
                "km_driven",
                "mileage_num",
                "engine_num",
                "max_power_num",
                "torque_num",
                "seats",
            ]
        ].describe().T,
        use_container_width=True
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8293ad;
        font-size:14px;
        padding:15px;
    ">
        AutoIQ | Intelligent Used Car Valuation System<br>
        Powered by Python, Streamlit, Plotly and Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)
