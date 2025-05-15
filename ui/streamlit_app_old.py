import streamlit as st
import pandas as pd
from utils import format_dataframe
from simulator.strategy_engine import load_initiatives, select_best_initiatives

# Load data
df = load_initiatives("data/initiatives.csv")
df = df.drop(columns=["Confidence Multiplier"])

st.set_page_config(layout="wide")
st.title("📊 Operational Strategy Simulator")

# === Constraints & Available Initiatives (Containerized Section) ===
with st.container():
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border:1px solid #e0e0e0;">
    <h3>🔧 Set Constraints & Choose Initiatives</h3>
    """, unsafe_allow_html=True)

    # Constraints input (now inside a structured section)
    budget = st.slider("💰 Max Budget ($)", 0, 300000, 150000, 10000)
    eng_days = st.slider("🛠️ Max Engineering Days", 0, 100, 40, 5)

    st.markdown("<h4>📝 Available Initiatives</h4>", unsafe_allow_html=True)

    # Checkbox Selection Grid
    selected_initiatives = []
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        col = cols[i % 2]
        with col:
            label = (
                f"**{row['Initiative']}**  \n"
                f"💸 ${row['Cost ($)']:,.0f} | 🛠 {row['Engineering Days']}d | 📈 +{row['ARR Impact (%)']}%"
            )
            if st.checkbox(label, value=True, key=row["Initiative"]):
                selected_initiatives.append(row["Initiative"])

    st.markdown("</div>", unsafe_allow_html=True)

# === Manual Selection Summary ===
manual_df = df[df["Initiative"].isin(selected_initiatives)].reset_index(drop=True)
manual_df = format_dataframe(manual_df)
total_cost = manual_df["Cost ($)"].replace({'\$': '', ',': ''}, regex=True).astype(float).sum()
total_days = manual_df["Engineering Days"].replace({',': ''}, regex=True).astype(int).sum()
total_arr = manual_df["Expected ARR Impact ($)"].replace({'\$': '', ',': ''}, regex=True).astype(float).sum()
total_roi = total_arr / total_cost if total_cost > 0 else 0  # Now works correctly!



with st.container():
    st.markdown("""
    <div style="background-color:#eaf7ea; padding:20px; border-radius:10px; border:1px solid #d6ebd5;">
    <h3>📋 Manual Selection Summary</h3>
    """, unsafe_allow_html=True)

    st.dataframe(manual_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cost", f"${total_cost:,.0f}", delta="✅ Within Budget" if total_cost <= budget else "❌ Over Budget")
        st.metric("Engineering Days", f"{total_days}", delta="✅ OK" if total_days <= eng_days else "❌ Too High")
    with col2:
        st.metric("Expected ARR Impact", f"${total_arr:,.0f}")
        st.metric("Total ROI", f"{total_roi:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

# === Optimized Selection Section ===
with st.container():
    st.markdown("""
    <div style="background-color:#f5f5fa; padding:20px; border-radius:10px; border:1px solid #cccccc;">
    <h3>🤖 Optimized Selection</h3>
    """, unsafe_allow_html=True)

    optimization_metric = st.radio("📈 Optimize for:", ["ARR Impact", "ROI"], horizontal=True)

    # Optimization logic
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

    st.dataframe(optimized_df, use_container_width=True)

    opt1, opt2 = st.columns(2)
    with opt1:
        st.metric("Optimized Cost", f"${opt_cost:,.0f}")
        st.metric("Optimized Eng. Days", f"{opt_days}")
    with opt2:
        st.metric("Optimized ARR Impact", f"${opt_arr:,.0f}")
        st.metric("Optimized ROI", f"{opt_roi:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)
