import pandas as pd
from config import BASE_ARR, CONFIDENCE_MULTIPLIER

def load_initiatives(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Confidence Multiplier"] = df["Confidence"].map(CONFIDENCE_MULTIPLIER)
    df["Expected ARR Impact ($)"] = BASE_ARR * (df["ARR Impact (%)"] / 100) * df["Confidence Multiplier"]
    df["ROI"] = (df["Expected ARR Impact ($)"] - df["Cost ($)"]) / df["Cost ($)"]
    return df

#def select_best_initiatives(df, budget, max_engineering_days):
 #   df_sorted = df.sort_values(by="ROI", ascending=False)
  #  selected = []
   # total_cost = total_days = 0

   # for _, row in df_sorted.iterrows():
    #    if total_cost + row["Cost ($)"] <= budget and total_days + row["Engineering Days"] <= max_engineering_days:
     #       selected.append(row)
      #      total_cost += row["Cost ($)"]
       #     total_days += row["Engineering Days"]

    #return pd.DataFrame(selected), total_cost, total_days
