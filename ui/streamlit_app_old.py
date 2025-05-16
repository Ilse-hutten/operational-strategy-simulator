import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
from ui.utils import format_dataframe
from simulator.strategy_engine import load_initiatives
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

# --- Compact CSS ---
st.markdown("""
<style>
body {
    background-color: #e3f0fa !important;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
[data-testid="stVerticalBlock"] {
    gap: 0.75rem !important;
}
[data-testid="column"] {
    gap: 0.75rem !important;
}
.metric-dashboard-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.metric-card {
    background: #f9f9fb;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: inset 0 0 0 1px #d8d8e0;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-label {
    color: #1976d2;
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 2px;
}
.metric-value-row {
    display: flex;
    align-items: center;
    gap: 0.4em;
}
.metric-value {
    color: #113355;
    font-size: 1.4rem;
    font-weight: 700;
}
.metric-delta {
    font-size: 1rem;
    font-weight: 500;
    margin-left: 0.5em;
}
.white-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 12px rgba(30,60,120,0.08);
}
.styled-table thead tr th {
    background-color: #1976d2 !important;
    color: #fff !important;
    font-weight: bold !important;
    font-size: 1.02em;
    padding: 6px 8px !important;
}
.styled-table tbody tr:nth-child(even) {
    background-color: #f8fbff !important;
}
.styled-table tbody tr:nth-child(odd) {
    background-color: #e8f0f9 !important;
}
.styled-table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    font-size: 0.98rem !important;
    border: 1px solid #c2d6eb;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Operational Strategy Simulator")

# --- Load Data ---
df = load_initiatives("data/initiatives.csv")
df = df.drop(columns=["Confidence Multiplier"])

# --- Layout: Constraints + Initiatives ---
col_constraints, col_divider, col_initiatives = st.columns([1, 0.02, 2], gap="small")


with col_constraints:
    st.markdown("### Constraints")
    budget = st.slider("💰 Max Budget ($)", 0, 300000, 150000, 10000)
    eng_days = st.slider("🛠️ Max Engineering Days", 0, 100, 40, 5)
    st.markdown("</div>", unsafe_allow_html=True)

with col_divider:
    st.markdown(
        """
        <div style="height: 100%; min-height: 400px; border-left: 2px solid #1976d2; margin: 0 auto;"></div>
        """,
        unsafe_allow_html=True,
    )

with col_initiatives:
    with st.container():
        # Manually inject a styled box background behind this container
        st.markdown('<div id="available-initiatives-box">', unsafe_allow_html=True)

        st.markdown("### 📝 Available Initiatives", unsafe_allow_html=True)

        selected_initiatives = []
        grid_cols = st.columns(2, gap="small")
        for i, (_, row) in enumerate(df.iterrows()):
            col = grid_cols[i % 2]
            with col:
                label = (
                    f"**{row['Initiative']}**  \n"
                    f"💸 ${row['Cost ($)']:,.0f} &nbsp;&nbsp; | &nbsp;&nbsp; 🛠 {row['Engineering Days']}d &nbsp;&nbsp; | &nbsp;&nbsp; 📈 +{row['ARR Impact (%)']}%"
                )
                if st.checkbox(label, value=True, key=row["Initiative"]):
                    selected_initiatives.append(row["Initiative"])

        st.markdown("</div>", unsafe_allow_html=True)





# Use regular DataFrame for calculations
manual_df = df[df["Initiative"].isin(selected_initiatives)].reset_index(drop=True)

# Do calculations on the raw manual_df
total_cost = manual_df["Cost ($)"].sum()
total_days = manual_df["Engineering Days"].sum()
total_arr = manual_df["Expected ARR Impact ($)"].sum()
total_roi = total_arr / total_cost if total_cost > 0 else 0

# Then format a display version separately
manual_df_formatted = format_dataframe(manual_df)

st.markdown(
    f"""
    <div style="background-color:#e3f0fa; padding:12px 8px 8px 8px; border-radius:10px; margin-bottom:12px;">
    <h4 style="color:#0068c9; margin-bottom:0.7em;">📋 Manual Selection Summary</h4>
    """,
    unsafe_allow_html=True,
)


st.write(manual_df_formatted, unsafe_allow_html=True)



m1, m2 = st.columns(2, gap="small")
with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value-row">
                <span class="metric-value">${total_cost:,.0f}</span>
                <span class="metric-delta" style="color:{'#1976d2' if total_cost <= budget else '#d21919'};">
                    {'✅' if total_cost <= budget else '❌'} {'Within Budget' if total_cost <= budget else 'Over Budget'}
                </span>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Engineering Days</div>
            <div class="metric-value-row">
                <span class="metric-value">{total_days}</span>
                <span class="metric-delta" style="color:{'#1976d2' if total_days <= eng_days else '#d21919'};">
                    {'✅' if total_days <= eng_days else '❌'} {'OK' if total_days <= eng_days else 'Too High'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Expected ARR Impact</div>
            <div class="metric-value-row">
                <span class="metric-value">${total_arr:,.0f}</span>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total ROI</div>
            <div class="metric-value-row">
                <span class="metric-value">{total_roi:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# --- Optimized Selection Section ---
st.markdown(
    f"""
    <div style="background-color:#e3f0fa; padding:12px 8px 8px 8px; border-radius:10px;">
    <h4 style="color:#0068c9; margin-bottom:0.7em;">🤖 Optimized Selection</h4>
    """,
    unsafe_allow_html=True,
)
optimization_metric = st.radio("📈 Optimize for:", ["ARR Impact", "ROI"], horizontal=True)
df["Score"] = df["Expected ARR Impact ($)"] if optimization_metric == "ARR Impact" else df["Expected ARR Impact ($)"] / df["Cost ($)"]
df_sorted = df.sort_values("Score", ascending=False)
opt_cost = opt_days = opt_arr = 0
optimized_initiatives = []
for _, row in df_sorted.iterrows():
    if (opt_cost + row["Cost ($)"] <= budget) and (opt_days + row["Engineering Days"] <= eng_days):
        optimized_initiatives.append(row)
        opt_cost += row["Cost ($)"]
        opt_days += row["Engineering Days"]
        opt_arr += row["Expected ARR Impact ($)"]
optimized_df = pd.DataFrame(optimized_initiatives).reset_index(drop=True)
opt_roi = opt_arr / opt_cost if opt_cost > 0 else 0

optimized_df_formatted = format_dataframe(optimized_df)
st.write(optimized_df_formatted, unsafe_allow_html=True)

# 👇 Begin white dashboard-style wrapper
st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)

o1, o2 = st.columns(2, gap="small")
with o1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Optimized Cost</div>
            <div class="metric-value-row">
                <span class="metric-value">${opt_cost:,.0f}</span>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Optimized Eng. Days</div>
            <div class="metric-value-row">
                <span class="metric-value">{opt_days}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with o2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Optimized ARR Impact</div>
            <div class="metric-value-row">
                <span class="metric-value">${opt_arr:,.0f}</span>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Optimized ROI</div>
            <div class="metric-value-row">
                <span class="metric-value">{opt_roi:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 👇 End dashboard box
#st.markdown('</div>', unsafe_allow_html=True)



# === Monte Carlo Simulation for Optimized Initiatives ===

# Std dev mapping from confidence level
confidence_to_std = {
    "High": 0.10,
    "Medium": 0.20,
    "Low": 0.40
}

# Add ARR_std based on confidence
optimized_df["ARR_std"] = optimized_df["Expected ARR Impact ($)"] * optimized_df["Confidence"].map(confidence_to_std)

# Simulation function
def run_simulation(subset_df, n_runs=1000):
    roi_results = []
    for _ in range(n_runs):
        total_arr = 0
        total_cost = 0
        for _, row in subset_df.iterrows():
            arr_sample = np.random.normal(loc=row["Expected ARR Impact ($)"], scale=row["ARR_std"])
            arr_sample = max(arr_sample, 0)
            total_arr += arr_sample
            total_cost += row["Cost ($)"]
        roi = total_arr / total_cost if total_cost > 0 else 0
        roi_results.append(roi)
    return roi_results

# --- UI Card Container ---
st.markdown("""
<div class="white-card">
    <h4 style="color:#0068c9; margin-bottom:1rem;">📈 Monte Carlo Simulation of ROI (Optimized)</h4>
""", unsafe_allow_html=True)

# Let user adjust number of simulation runs
n_runs = st.slider("Number of Simulations", 100, 5000, 1000, step=100)

# Run simulation
roi_sim = run_simulation(optimized_df, n_runs=n_runs)

# Plot
fig, ax = plt.subplots(figsize=(8, 3.8))
counts, bins, _ = ax.hist(roi_sim, bins=30, color='#a7d3f5', edgecolor='#336699', alpha=0.7, density=True)

kde = gaussian_kde(roi_sim)
x_vals = np.linspace(min(roi_sim), max(roi_sim), 200)
ax.plot(x_vals, kde(x_vals), color='#0a369d', linewidth=2, label='KDE')

ax.set_title("Simulated ROI Distribution", fontsize=13, fontweight='bold', color='#003366')
ax.set_xlabel("ROI", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.tick_params(axis='both', labelsize=10)
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=10)

st.pyplot(fig)

# Percentile summary
p10, p50, p90 = np.percentile(roi_sim, [10, 50, 90])
st.markdown(
    f"""
    <div style="margin-top: 0.5rem; font-size: 0.95rem; color: #444;">
    <b>Percentile Summary:</b><br>
    10th percentile ROI: <b>{p10:.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
    Median ROI: <b>{p50:.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
    90th percentile ROI: <b>{p90:.2f}</b>
    </div>
    """, unsafe_allow_html=True
)

# Close white card container
st.markdown("</div>", unsafe_allow_html=True)
