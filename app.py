import streamlit as st
import pandas as pd
import numpy as np
import psutil
import joblib
import time
import os
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI System Failure Prediction",
    page_icon="🖥️",
    layout="wide"
)

# ============================================================
# FILE NAMES
# ============================================================

FAILURE_DATASET = "failure_prediction.csv"
REALTIME_DATASET = "realtime_system_data.csv"

MODEL_FILE = "failure_prediction_model.pkl"
ENCODER_FILE = "failure_risk_encoder.pkl"

# ============================================================
# TITLE
# ============================================================

st.title("🖥️ AI-Powered System Crash & Hardware Stress Prediction")

st.write(
    "Real-time laptop monitoring and AI-based hardware "
    "failure prediction using custom datasets."
)

st.caption("🔴 LIVE MONITORING")

st.divider()

# ============================================================
# LOAD DATASETS
# ============================================================

failure_df = None
realtime_df = None

if os.path.exists(FAILURE_DATASET):
    try:
        failure_df = pd.read_csv(FAILURE_DATASET)
    except Exception as e:
        st.error(f"Error reading {FAILURE_DATASET}: {e}")

if os.path.exists(REALTIME_DATASET):
    try:
        realtime_df = pd.read_csv(REALTIME_DATASET)
    except Exception as e:
        st.error(f"Error reading {REALTIME_DATASET}: {e}")

# ============================================================
# LOAD ML MODEL
# ============================================================

model = None
encoder = None

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model: {e}")

if os.path.exists(ENCODER_FILE):
    try:
        encoder = joblib.load(ENCODER_FILE)
    except Exception as e:
        st.error(f"Error loading encoder: {e}")

# ============================================================
# LIVE LAPTOP DATA
# ============================================================

cpu = psutil.cpu_percent(interval=1)

ram = psutil.virtual_memory().percent

disk = psutil.disk_usage("C:\\").percent

process_count = len(psutil.pids())

# ============================================================
# NETWORK DATA
# ============================================================

network = psutil.net_io_counters()

network_total_mb = (
    network.bytes_sent + network.bytes_recv
) / (1024 * 1024)

# Network usage percentage
# Based on current network traffic relative to a practical
# reference level for dashboard purposes.

network_usage_percent = min(
    (network_total_mb / 5000) * 100,
    100
)

# ============================================================
# ACTIVE JOBS
# ============================================================

active_jobs = process_count

# ============================================================
# IDLE TIME ESTIMATION
# ============================================================

# Higher CPU usage means lower idle time.
# This is a dashboard-compatible derived metric.

idle_time_minutes = max(
    0,
    round((100 - cpu) * 5, 2)
)

# ============================================================
# POWER CONSUMPTION ESTIMATION
# ============================================================

# Approximate system power score based on CPU and RAM load.
# This is an estimated value, not a hardware power sensor.

power_consumption_watts = round(
    80
    + (cpu * 2.2)
    + (ram * 1.1),
    2
)

# ============================================================
# TEMPERATURE ESTIMATION
# ============================================================

# Approximate temperature based on CPU utilization.
# If a real temperature sensor is available, it can replace this.

temperature_celsius = round(
    35 + (cpu * 0.35),
    2
)

# ============================================================
# HARDWARE STRESS SCORE
# ============================================================

stress_score = (
    cpu * 0.40
    + ram * 0.35
    + disk * 0.25
)

stress_score = round(
    min(stress_score, 100),
    2
)

# ============================================================
# LIVE LAPTOP SECTION
# ============================================================

st.subheader("🖥️ Live Laptop Monitoring")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "CPU Usage",
        f"{cpu:.1f}%"
    )

with col2:
    st.metric(
        "RAM Usage",
        f"{ram:.1f}%"
    )

with col3:
    st.metric(
        "Disk Usage",
        f"{disk:.1f}%"
    )

with col4:
    st.metric(
        "Network",
        f"{network_total_mb:.2f} MB"
    )

with col5:
    st.metric(
        "Active Processes",
        process_count
    )

st.divider()

# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader("🩺 Current System Status")

if cpu >= 90 or ram >= 90:

    st.error(
        "🔴 HIGH SYSTEM STRESS"
    )

elif cpu >= 70 or ram >= 70:

    st.warning(
        "🟠 MEDIUM SYSTEM STRESS"
    )

else:

    st.success(
        "🟢 SYSTEM NORMAL"
    )

# ============================================================
# STRESS SCORE
# ============================================================

st.subheader("⚡ Hardware Stress Score")

st.progress(
    int(stress_score)
)

st.write(
    f"Hardware Stress Score: "
    f"**{stress_score} / 100**"
)

# ============================================================
# AI FAILURE PREDICTION
# ============================================================

st.divider()

st.subheader("🤖 AI Failure Prediction")

prediction = "Unknown"
confidence = 0.0

if model is not None:

    try:

        # ====================================================
        # EXACT FEATURES USED BY TRAINED MODEL
        # ====================================================

        live_input = pd.DataFrame(
            [
                {
                    "cpu_usage_percent": cpu,
                    "ram_usage_percent": ram,
                    "disk_usage_percent": disk,
                    "network_usage_percent": network_usage_percent,
                    "active_jobs": active_jobs,
                    "idle_time_minutes": idle_time_minutes,
                    "power_consumption_watts": power_consumption_watts,
                    "temperature_celsius": temperature_celsius
                }
            ]
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        prediction_encoded = model.predict(
            live_input
        )[0]

        # ====================================================
        # DECODE PREDICTION
        # ====================================================

        if encoder is not None:

            try:

                prediction = encoder.inverse_transform(
                    [prediction_encoded]
                )[0]

            except Exception:

                prediction = str(
                    prediction_encoded
                )

        else:

            prediction = str(
                prediction_encoded
            )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                live_input
            )[0]

            confidence = float(
                max(probabilities) * 100
            )

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        prediction_text = str(
            prediction
        ).lower()

        if prediction_text == "high":

            st.error(
                "🔴 HIGH FAILURE RISK"
            )

        elif prediction_text == "medium":

            st.warning(
                "🟠 MEDIUM FAILURE RISK"
            )

        elif prediction_text == "low":

            st.success(
                "🟢 LOW FAILURE RISK"
            )

        else:

            st.info(
                f"AI Prediction: {prediction}"
            )

        st.metric(
            "AI Confidence",
            f"{confidence:.1f}%"
        )

        # ====================================================
        # SHOW DATA SENT TO MODEL
        # ====================================================

        with st.expander(
            "🔍 View Data Sent to AI Model"
        ):

            st.dataframe(
                live_input,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            "❌ AI prediction could not be generated."
        )

        st.code(
            str(e)
        )

else:

    st.warning(
        "⚠️ failure_prediction_model.pkl "
        "not found."
    )

# ============================================================
# REALTIME DATASET
# ============================================================

st.divider()

st.subheader(
    "📊 Real-Time System Dataset"
)

if realtime_df is not None:

    st.write(
        f"Total Records: **{len(realtime_df)}**"
    )

    st.dataframe(
        realtime_df,
        use_container_width=True,
        height=350
    )

    # ========================================================
    # REALTIME DATASET ANALYTICS
    # ========================================================

    st.subheader(
        "📈 Real-Time Dataset Analytics"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        if "cpu_usage_percent" in realtime_df.columns:

            st.metric(
                "Average CPU",
                f"{realtime_df['cpu_usage_percent'].mean():.2f}%"
            )

    with c2:

        if "ram_usage_percent" in realtime_df.columns:

            st.metric(
                "Average RAM",
                f"{realtime_df['ram_usage_percent'].mean():.2f}%"
            )

    with c3:

        if "disk_usage_percent" in realtime_df.columns:

            st.metric(
                "Average Disk",
                f"{realtime_df['disk_usage_percent'].mean():.2f}%"
            )

    with c4:

        if "active_processes" in realtime_df.columns:

            st.metric(
                "Average Processes",
                f"{realtime_df['active_processes'].mean():.0f}"
            )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    if "failure_risk" in realtime_df.columns:

        st.write(
            "### 🚦 Failure Risk Distribution"
        )

        risk_counts = (
            realtime_df["failure_risk"]
            .value_counts()
        )

        st.bar_chart(
            risk_counts
        )

# ============================================================
# FAILURE PREDICTION DATASET
# ============================================================

st.divider()

st.subheader(
    "📚 AI Training Dataset"
)

if failure_df is not None:

    st.write(
        f"Total Training Records: "
        f"**{len(failure_df)}**"
    )

    st.dataframe(
        failure_df,
        use_container_width=True,
        height=350
    )

    # ========================================================
    # FAILURE DATASET RISK
    # ========================================================

    if "failure_risk" in failure_df.columns:

        st.subheader(
            "📊 Training Dataset Risk Distribution"
        )

        failure_risk_counts = (
            failure_df["failure_risk"]
            .value_counts()
        )

        st.bar_chart(
            failure_risk_counts
        )

    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    st.subheader(
        "📈 Training Dataset Statistics"
    )

    numeric_columns = (
        failure_df
        .select_dtypes(include=np.number)
        .columns
    )

    if len(numeric_columns) > 0:

        st.dataframe(
            failure_df[
                numeric_columns
            ].describe(),
            use_container_width=True
        )

else:

    st.warning(
        "⚠️ failure_prediction.csv not found."
    )

# ============================================================
# LIVE VS TRAINING DATASET
# ============================================================

st.divider()

st.subheader(
    "🔍 Live Laptop vs Training Dataset"
)

if failure_df is not None:

    comparison_data = []

    comparison_data.append(
        {
            "Metric": "CPU Usage (%)",
            "Live Laptop": cpu,
            "Dataset Average":
                failure_df[
                    "cpu_usage_percent"
                ].mean()
        }
    )

    comparison_data.append(
        {
            "Metric": "RAM Usage (%)",
            "Live Laptop": ram,
            "Dataset Average":
                failure_df[
                    "ram_usage_percent"
                ].mean()
        }
    )

    comparison_data.append(
        {
            "Metric": "Disk Usage (%)",
            "Live Laptop": disk,
            "Dataset Average":
                failure_df[
                    "disk_usage_percent"
                ].mean()
        }
    )

    comparison_df = pd.DataFrame(
        comparison_data
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.divider()

st.subheader(
    "💻 System Information"
)

memory = psutil.virtual_memory()

cpu_frequency = psutil.cpu_freq()

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "CPU Cores",
        psutil.cpu_count()
    )

with c2:

    st.metric(
        "Total RAM",
        f"{memory.total / (1024**3):.2f} GB"
    )

with c3:

    st.metric(
        "Available RAM",
        f"{memory.available / (1024**3):.2f} GB"
    )

with c4:

    if cpu_frequency:

        st.metric(
            "CPU Frequency",
            f"{cpu_frequency.current:.0f} MHz"
        )

# ============================================================
# LIVE HARDWARE DETAILS
# ============================================================

st.divider()

st.subheader(
    "⚙️ Live Hardware Details"
)

details = pd.DataFrame(
    {
        "Parameter": [
            "CPU Usage",
            "RAM Usage",
            "Disk Usage",
            "Network Usage",
            "Active Jobs",
            "Idle Time",
            "Power Consumption",
            "Temperature"
        ],
        "Current Value": [
            f"{cpu:.2f} %",
            f"{ram:.2f} %",
            f"{disk:.2f} %",
            f"{network_usage_percent:.2f} %",
            active_jobs,
            f"{idle_time_minutes:.2f} minutes",
            f"{power_consumption_watts:.2f} W",
            f"{temperature_celsius:.2f} °C"
        ]
    }
)

st.dataframe(
    details,
    use_container_width=True
)

# ============================================================
# AI RECOMMENDATIONS
# ============================================================

st.divider()

st.subheader(
    "🤖 AI Recommendations"
)

recommendations = []

if cpu >= 80:

    recommendations.append(
        "⚠️ CPU usage is high. "
        "Reduce heavy background workloads."
    )

if ram >= 80:

    recommendations.append(
        "⚠️ RAM usage is high. "
        "Close unnecessary applications."
    )

if disk >= 85:

    recommendations.append(
        "⚠️ Disk usage is high. "
        "Consider cleaning storage."
    )

if process_count >= 250:

    recommendations.append(
        "⚠️ Large number of active processes detected."
    )

if stress_score >= 70:

    recommendations.append(
        "🔴 High hardware stress detected."
    )

elif stress_score >= 40:

    recommendations.append(
        "🟠 Moderate hardware stress detected."
    )

else:

    recommendations.append(
        "🟢 System hardware is currently operating normally."
    )

for item in recommendations:

    st.write(item)

# ============================================================
# DOWNLOAD CURRENT PREDICTION
# ============================================================

st.divider()

st.subheader(
    "📥 Current Prediction Report"
)

report_df = pd.DataFrame(
    [
        {
            "Time": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "CPU (%)": round(cpu, 2),
            "RAM (%)": round(ram, 2),
            "Disk (%)": round(disk, 2),
            "Network (MB)": round(
                network_total_mb,
                2
            ),
            "Processes": process_count,
            "Risk": prediction,
            "Confidence (%)": round(
                confidence,
                2
            ),
            "Stress Score": stress_score
        }
    ]
)

st.dataframe(
    report_df,
    use_container_width=True
)

csv_data = report_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Current Prediction CSV",
    data=csv_data,
    file_name="system_failure_prediction_report.csv",
    mime="text/csv"
)

# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(2)

st.rerun()