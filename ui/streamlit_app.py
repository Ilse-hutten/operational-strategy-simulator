import streamlit as st
import pandas as pd
from ui.utils import format_dataframe
from simulator.strategy_engine import load_initiatives
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")

# --- CSS ---
st.markdown("""
<style>
body { background-color: #e3f0fa !important; }
.block-container { padding-top: 1.8rem; padding-bottom: 0.5rem; padding-left: 2rem; padding-right: 2rem; }
[data-testid="stVerticalBlock"], [data-testid="column"] { gap: 0.5rem !important; }
.metric-dashboard-box { background: #ffffff; border-radius: 14px; padding: 16px 18px; margin-bottom: 1.2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
.metric-card { background: #f0f7ff; border-radius: 10px; padding: 10px 12px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(30,60,120,0.06); display: flex; flex-direction: column; justify-content: center; min-height: 56px; }
.metric-label { color: #1976d2; font-size: 1.05rem; font-weight: 600; margin-bottom: 1px; }
.metric-value-row { display: flex; align-items: center; gap: 0.5em; }
.metric-value { color: #113355; font-size: 1.2rem; font-weight: 700; }
.metric-delta { font-size: 1.05rem; font-weight: 500; margin-left: 0.6em; }
.white-card { background: #fff; border-radius: 10px; padding: 12px 16px 8px 16px; margin-bottom: 0.5rem; box-shadow: 0 2px 8px rgba(30,60,120,0.07); }
.styled-table thead tr th { background-color: #1976d2 !important; color: #fff !important; font-weight: bold !important; font-size: 1.02em; padding: 6px 8px !important; }
.styled-table tbody tr:nth-child(even) { background-color: #f8fbff !important; }
.styled-table tbody tr:nth-child(odd) { background-color: #e3f0fa !important; }
.styled-table { border-collapse: separate !important; border-spacing: 0 !important; width: 100% !important; border-radius: 8px !important; overflow: hidden !important; font-size: 0.97rem !important; border: 1px solid #c2d6eb; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Operational Strategy Simulator")

# --- LOAD DATA ---
df_base = load_initiatives("data/initiatives.csv").drop(columns=["Confidence Multiplier"])
df_manual = df_base.copy()
df_opt = df_base.copy()

# --- CONSTRAINTS ---
col_constraints, col_initiatives = st.columns([1, 2], gap="small")
with col_constraints:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown("### Constraints")
    budget = st.slider("💰 Max Budget ($)", 0, 300000, 150000, 10000)
    eng_days = st.slider("🛠️ Max Engineering Days", 0, 100, 40, 5)

# --- MANUAL INITIATIVES SELECTION ---
with col_initiatives:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Available Initiatives", unsafe_allow_html=True)
    selected_initiatives = []
    grid_cols = st.columns(2, gap="small")
    for i, (_, row) in enumerate(df_manual.iterrows()):
        col = grid_cols[i % 2]
        with col:
            label = (
                f"**{row['Initiative']}**  \n"
                f"💸 ${row['Cost ($)']:,.0f} &nbsp;|&nbsp; 🛠 {row['Engineering Days']}d &nbsp;|&nbsp; 📈 +{row['ARR Impact (%)']}%"
            )
            if st.checkbox(label, value=True, key=row["Initiative"]):
                selected_initiatives.append(row["Initiative"])
    st.markdown("</div>", unsafe_allow_html=True)

# --- MANUAL SELECTION SUMMARY ---
manual_df = df_manual[df_manual["Initiative"].isin(selected_initiatives)].reset_index(drop=True)
manual_df_formatted = format_dataframe(manual_df)
total_cost = manual_df["Cost ($)"].sum()
total_days = manual_df["Engineering Days"].sum()
total_arr = manual_df["Expected ARR Impact ($)"].sum()
total_roi = total_arr / total_cost if total_cost > 0 else 0

st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
st.markdown("#### 📋 Manual Selection Summary", unsafe_allow_html=True)
st.write(manual_df_formatted.set_table_attributes('class="styled-table"'), unsafe_allow_html=True)

m1, m2 = st.columns(2, gap="small")
with m1:
    st.markdown(f"""
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
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
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
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- OPTIMIZED SELECTION ---
st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
st.markdown("#### 🤖 Optimized Selection", unsafe_allow_html=True)
optimization_metric = st.radio("📈 Optimize for:", ["ARR Impact", "ROI"], horizontal=True)
df_opt["Score"] = df_opt["Expected ARR Impact ($)"] if optimization_metric == "ARR Impact" else df_opt["Expected ARR Impact ($)"] / df_opt["Cost ($)"]
df_sorted = df_opt.sort_values("Score", ascending=False)

opt_cost = opt_days = opt_arr = 0
optimized_initiatives = []
for _, row in df_sorted.iterrows():
    if (opt_cost + row["Cost ($)"] <= budget) and (opt_days + row["Engineering Days"] <= eng_days):
        optimized_initiatives.append(row)
        opt_cost += row["Cost ($)"]
        opt_days += row["Engineering Days"]
        opt_arr += row["Expected ARR Impact ($)"]

optimized_df = pd.DataFrame(optimized_initiatives).reset_index(drop=True)
optimized_df_formatted = format_dataframe(optimized_df)
opt_roi = opt_arr / opt_cost if opt_cost > 0 else 0

st.write(optimized_df_formatted.set_table_attributes('class="styled-table"'), unsafe_allow_html=True)

o1, o2 = st.columns(2, gap="small")
with o1:
    st.markdown(f"""
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
    """, unsafe_allow_html=True)
with o2:
    st.markdown(f"""
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
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# === METRICS + MONTE CARLO ===
dashboard_col1, dashboard_col2 = st.columns([1, 1], gap="small")

with dashboard_col1:
    st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
    st.markdown("#### 🚦 Performance Metrics", unsafe_allow_html=True)
    num_selected = len(optimized_df)
    avg_cost = optimized_df["Cost ($)"].mean() if num_selected > 0 else 0
    avg_roi = (optimized_df["Expected ARR Impact ($)"] / optimized_df["Cost ($)"]).mean() if num_selected > 0 else 0
    total_arr = optimized_df["Expected ARR Impact ($)"].sum()
    mcol1, mcol2 = st.columns(2, gap="small")
    with mcol1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Initiatives Selected</div>
                <div class="metric-value-row">
                    <span class="metric-value">{num_selected}</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg. Initiative Cost</div>
                <div class="metric-value-row">
                    <span class="metric-value">${avg_cost:,.0f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg. ROI</div>
                <div class="metric-value-row">
                    <span class="metric-value">{avg_roi:.2f}</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Expected ARR Impact</div>
                <div class="metric-value-row">
                    <span class="metric-value">${total_arr:,.0f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with dashboard_col2:
    st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
    st.markdown("#### 📈 Monte Carlo Simulation (Optimized)", unsafe_allow_html=True)
    st.markdown("""
<p style="font-size: 0.95rem; color: #444;">
The Monte Carlo simulator estimates the <b>Return on Investment (ROI)</b> distribution by repeatedly sampling possible outcomes for each initiative.
Each run introduces randomness in ARR impact (based on confidence levels) and, in <b>Bad</b> scenarios, adds unexpected <b>cost overruns</b>.
This allows you to see how your optimized selection might perform under uncertainty.
</p>

<ul style="font-size: 0.9rem; color: #444; margin-top: -0.5rem;">
  <li><b>Good:</b> Low ARR uncertainty, no cost inflation</li>
  <li><b>Neutral:</b> Baseline uncertainty, no cost inflation</li>
  <li><b>Bad:</b> High ARR uncertainty + 10–30% cost shock</li>
</ul>
""", unsafe_allow_html=True)


    # --- Scenario Selection ---
    scenario = st.selectbox("📉 Scenario", ["Good", "Neutral", "Bad"], index=1)
    scenario_multipliers = {"Good": 0.7, "Neutral": 1.0, "Bad": 1.5}
    scenario_factor = scenario_multipliers[scenario]

    confidence_to_std = {"High": 0.10, "Medium": 0.20, "Low": 0.40}

    if not optimized_df.empty:
        # Apply ARR uncertainty per confidence level and scenario
        optimized_df["ARR_std"] = (
            optimized_df["Expected ARR Impact ($)"]
            * optimized_df["Confidence"].map(confidence_to_std)
            * scenario_factor
        )

        def run_simulation(subset_df, n_runs=1000):
            roi_results = []
            for _ in range(n_runs):
                total_arr = 0
                total_cost = 0
                for _, row in subset_df.iterrows():
                    # Simulate ARR with uncertainty
                    arr_sample = np.random.normal(loc=row["Expected ARR Impact ($)"], scale=row["ARR_std"])
                    arr_sample = max(arr_sample, 0)

                    # Simulate cost (shock if in Bad scenario)
                    cost = row["Cost ($)"]
                    if scenario == "Bad":
                        cost *= np.random.uniform(1.1, 1.3)  # 10–30% inflation
                    total_arr += arr_sample
                    total_cost += cost

                roi = total_arr / total_cost if total_cost > 0 else 0
                roi_results.append(roi)
            return roi_results

        n_runs = st.slider("Number of Simulations", 100, 5000, 1000, step=100)
        roi_sim = run_simulation(optimized_df, n_runs=n_runs)

        # --- Plot Distribution ---
        fig, ax = plt.subplots(figsize=(6, 3))
        counts, bins, _ = ax.hist(roi_sim, bins=30, color='#a7d3f5', edgecolor='#336699', alpha=0.7, density=True)
        kde = gaussian_kde(roi_sim)
        x_vals = np.linspace(min(roi_sim), max(roi_sim), 200)
        ax.plot(x_vals, kde(x_vals), color='#0a369d', linewidth=2, label='KDE')
        ax.set_title(f"Simulated ROI Distribution ({scenario} Scenario)", fontsize=12, fontweight='bold', color='#003366')
        ax.set_xlabel("ROI", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)

        # --- Percentile Summary ---
        p10, p50, p90 = np.percentile(roi_sim, [10, 50, 90])
        st.markdown(f"""
            <div style="margin-top: 0.5rem; font-size: 0.95rem; color: #444;">
            <b>Percentile Summary:</b><br>
            10th percentile ROI: <b>{p10:.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            Median ROI: <b>{p50:.2f}</b> &nbsp;&nbsp;|&nbsp;&nbsp;
            90th percentile ROI: <b>{p90:.2f}</b>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No optimized initiatives selected for simulation.")
    st.markdown("</div>", unsafe_allow_html=True)
