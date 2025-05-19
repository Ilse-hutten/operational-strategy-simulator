# Operational Strategy Simulator

A Streamlit app to simulate and optimize strategic initiatives based on cost, engineering constraints, and ARR/ROI impact. It enables users to select initiatives manually or use an automated knapsack-based optimizer, and visualize expected ROI distributions under uncertainty using Monte Carlo simulations.

<img src="visuals/initiative_performance.png" alt="relationships" width="700"/>

---

## Live Demo

🔗 Try the app here:  
[https://operational-strategy-simulator-hxdqqzqz8i6s5kagkvryc3.streamlit.app](https://operational-strategy-simulator-hxdqqzqz8i6s5kagkvryc3.streamlit.app)

---

## Features

-  **Interactive budget and engineering constraints**
-  **Manual selection of initiatives with performance metrics**
-  **Optimized selection via linear programming (ARR or ROI)**
-  **Monte Carlo simulation to evaluate risk-adjusted ROI**
-  **Scenario modeling (Good / Neutral / Bad)**
-  **Clean UI with professional layout and summaries**

---

### Optimization Logic

- Each initiative is treated as a **binary decision**:
  - `1` = selected  
  - `0` = not selected

- The model **maximizes** either:
  - **ARR Impact** → total expected ARR from selected initiatives
  - **ROI (Net Gain)** → total ARR minus total cost from selected initiatives

- **Objective functions**:
  - Maximize ARR: sum of `(ARR Impact × selected)` across initiatives
  - Maximize ROI: sum of `((ARR Impact - Cost) × selected)` across initiatives

- **Subject to constraints**:
  - Total **cost** of selected initiatives ≤ available **budget**
  - Total **engineering days** required ≤ available **engineering capacity**

- Solved using a **binary knapsack optimization** model via `pulp`


---


### Monte Carlo ROI Simulation
<img src="visuals/monte_carlo_roi.png" alt="Monte Carlo ROI" width="400"/>

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/operational-strategy-simulator.git
cd operational-strategy-simulator
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Launch the app
```bash
streamlit run ui/streamlit_app.py
```
