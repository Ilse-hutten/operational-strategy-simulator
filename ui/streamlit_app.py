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
# --- Custom CSS ---
st.markdown("""
<style>
body { background-color: #e3f0fa !important; }
.block-container { padding: 1.8rem 2rem; }
[data-testid="stVerticalBlock"] { gap: 1rem !important; }
[data-testid="column"] { gap: 1rem !important; }

.metric-dashboard-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.metric-card {
    background: #f4f8fd;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
    box-shadow: inset 0 0 0 1px #d0dce8;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-label {
    color: #1976d2;
    font-size: 1.02rem;
    font-weight: 600;
    margin-bottom: 3px;
}
.metric-value-row {
    display: flex;
    align-items: center;
    gap: 0.4em;
}
.metric-value {
    color: #113355;
    font-size: 1.3rem;
    font-weight: 700;
}
.metric-delta {
    font-size: 0.95rem;
    font-weight: 500;
    margin-left: 0.5em;
}
.white-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(30,60,120,0.06);
}
.styled-table thead tr th {
    background-color: #1976d2 !important;
    color: #fff !important;
    font-weight: bold !important;
    font-size: 0.98em;
    padding: 4px 6px !important;
}
.styled-table tbody tr:nth-child(even) { background-color: #f8fbff !important; }
.styled-table tbody tr:nth-child(odd) { background-color: #e6f0f9 !important; }
.styled-table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    font-size: 0.95rem !important;
    border: 1px solid #c2d6eb;
}
</style>
""", unsafe_allow_html=True)

st.title("\U0001F4CA Operational Strategy Simulator")

# --- LOAD DATA ---
df_base = load_initiatives("data/initiatives.csv").drop(columns=["Confidence Multiplier"])
df_manual = df_base.copy()
df_opt = df_base.copy()

# --- CONSTRAINTS ---
col_constraints, col_initiatives = st.columns([1, 2], gap="small")
with col_constraints:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown("### Constraints")
    budget = st.slider("\U0001F4B0 Max Budget ($)", 0, 300000, 150000, 10000)
    eng_days = st.slider("\U0001F6E0️ Max Engineering Days", 0, 100, 40, 5)

# --- MANUAL INITIATIVES SELECTION ---
with col_initiatives:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown("### \U0001F4DD Available Initiatives", unsafe_allow_html=True)
    selected_initiatives = []
    grid_cols = st.columns(2, gap="small")
    for i, (_, row) in enumerate(df_manual.iterrows()):
        col = grid_cols[i % 2]
        with col:
            label = (
                f"**{row['Initiative']}**  \n"
                f"\U0001F4B8 ${row['Cost ($)']:,.0f} &nbsp;|&nbsp; \U0001F6E0 {row['Engineering Days']}d &nbsp;|&nbsp; \U0001F4C8 +{row['ARR Impact (%)']}%"
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
total_roi = (total_arr - total_cost) / total_cost if total_cost > 0 else 0

manual_col_df, manual_col_metrics = st.columns([2, 1], gap="small")
with manual_col_df:
    st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
    st.markdown("#### \U0001F4CB Manual Selection Summary", unsafe_allow_html=True)
    st.write(manual_df_formatted.to_html(), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with manual_col_metrics:
    st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
    st.markdown("#### \U0001F9EE Manual Performance Metrics", unsafe_allow_html=True)
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Cost</div><div class="metric-value-row">
        <span class="metric-value">${total_cost:,.0f}</span>
        <span class="metric-delta" style="color:{'#1976d2' if total_cost <= budget else '#d21919'};">
        {'✅' if total_cost <= budget else '❌'} {'Within Budget' if total_cost <= budget else 'Over Budget'}
        </span></div></div>
        <div class="metric-card"><div class="metric-label">Engineering Days</div><div class="metric-value-row">
        <span class="metric-value">{total_days}</span>
        <span class="metric-delta" style="color:{'#1976d2' if total_days <= eng_days else '#d21919'};">
        {'✅' if total_days <= eng_days else '❌'} {'OK' if total_days <= eng_days else 'Too High'}
        </span></div></div>
        <div class="metric-card"><div class="metric-label">Expected ARR Impact</div><div class="metric-value-row">
        <span class="metric-value">${total_arr:,.0f}</span>
        </div></div>
        <div class="metric-card"><div class="metric-label">Total ROI</div><div class="metric-value-row">
        <span class="metric-value">{total_roi:.2f}</span>
        </div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- OPTIMIZED SELECTION ---
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

opt_col_df, opt_col_metrics = st.columns([2, 1], gap="small")

with opt_col_df:
    st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
    st.markdown("#### \U0001F916 Optimized Selection", unsafe_allow_html=True)
    optimization_metric = st.radio("\U0001F4C8 Optimize for:", ["ARR Impact", "ROI"], horizontal=True)

    # Define optimization model
    model = LpProblem("Initiative_Selection", LpMaximize)
    x = [LpVariable(f"x{i}", cat=LpBinary) for i in range(len(df_opt))]

    # Objective function
    if optimization_metric == "ARR Impact":
        model += lpSum(df_opt["Expected ARR Impact ($)"].iloc[i] * x[i] for i in range(len(df_opt)))
    else:  # ROI = (ARR - Cost) / Cost --> Maximize net gain
        model += lpSum((df_opt["Expected ARR Impact ($)"].iloc[i] - df_opt["Cost ($)"].iloc[i]) * x[i] for i in range(len(df_opt)))

    # Constraints
    model += lpSum(df_opt["Cost ($)"].iloc[i] * x[i] for i in range(len(df_opt))) <= budget
    model += lpSum(df_opt["Engineering Days"].iloc[i] * x[i] for i in range(len(df_opt))) <= eng_days

    # Solve model
    model.solve(PULP_CBC_CMD(msg=0))
    selected = [i for i in range(len(df_opt)) if x[i].value() == 1]

    optimized_df = df_opt.iloc[selected].reset_index(drop=True)
    opt_cost = optimized_df["Cost ($)"].sum()
    opt_days = optimized_df["Engineering Days"].sum()
    opt_arr = optimized_df["Expected ARR Impact ($)"].sum()
    opt_roi = (opt_arr - opt_cost) / opt_cost if opt_cost > 0 else 0
    optimized_df_formatted = format_dataframe(optimized_df)

    st.dataframe(optimized_df_formatted, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with opt_col_metrics:
    st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
    st.markdown("#### \U0001F9EE Optimized Performance Metrics", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="metric-card"><div class="metric-label">Optimized Cost</div><div class="metric-value-row">
            <span class="metric-value">${opt_cost:,.0f}</span>
        </div></div>
        <div class="metric-card"><div class="metric-label">Optimized Eng. Days</div><div class="metric-value-row">
            <span class="metric-value">{opt_days}</span>
        </div></div>
        <div class="metric-card"><div class="metric-label">Optimized ARR Impact</div><div class="metric-value-row">
            <span class="metric-value">${opt_arr:,.0f}</span>
        </div></div>
        <div class="metric-card"><div class="metric-label">Optimized ROI</div><div class="metric-value-row">
            <span class="metric-value">{opt_roi:.2f}</span>
        </div></div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)




# === MONTE CARLO SIMULATION ===
mc_col_summary, mc_col_plot = st.columns([1, 2], gap="small")

with mc_col_summary:
    st.markdown('<div class="metric-dashboard-box">', unsafe_allow_html=True)
    st.markdown("#### \U0001F4C9 Monte Carlo Simulation (Optimized)", unsafe_allow_html=True)
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

with mc_col_plot:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    scenario = st.selectbox("\U0001F4C9 Scenario", ["Good", "Neutral", "Bad"], index=1)
    scenario_multipliers = {"Good": 0.7, "Neutral": 1.0, "Bad": 1.5}
    scenario_factor = scenario_multipliers[scenario]

    confidence_to_std = {"High": 0.10, "Medium": 0.20, "Low": 0.40}

    if not optimized_df.empty:
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
                    arr_sample = np.random.normal(loc=row["Expected ARR Impact ($)"], scale=row["ARR_std"])
                    arr_sample = max(arr_sample, 0)
                    cost = row["Cost ($)"]
                    if scenario == "Bad":
                        cost *= np.random.uniform(1.1, 1.3)
                    total_arr += arr_sample
                    total_cost += cost
                roi = (total_arr - total_cost) / total_cost if total_cost > 0 else 0
                roi_results.append(roi)
            return roi_results

        n_runs = st.slider("Number of Simulations", 100, 5000, 1000, step=100)
        roi_sim = run_simulation(optimized_df, n_runs=n_runs)

        fig, ax = plt.subplots(figsize=(3, 2))
        counts, bins, _ = ax.hist(roi_sim, bins=30, color='#a7d3f5', edgecolor='#336699', alpha=0.7, density=True)
        kde = gaussian_kde(roi_sim)
        x_vals = np.linspace(min(roi_sim), max(roi_sim), 200)
        ax.plot(x_vals, kde(x_vals), color='#0a369d', linewidth=2, label='KDE')
        ax.set_title(f"Simulated ROI Distribution ({scenario} Scenario)", fontsize=9, fontweight='bold', color='#003366')
        ax.set_xlabel("ROI", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.image(buf, use_column_width=False, width=280)


        p10, p50, p90 = np.percentile(roi_sim, [10, 50, 90])
        st.markdown("""
        <div class="metric-dashboard-box">
        <h4 style="color:#0068c9; margin-bottom:1rem;">📊 Percentile Summary</h4>
        <div style="font-size: 0.95rem; color: #444;">
        <b>10th percentile ROI:</b> {:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Median ROI:</b> {:.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>90th percentile ROI:</b> {:.2f}
        </div>
        </div>
        """.format(p10, p50, p90), unsafe_allow_html=True)
