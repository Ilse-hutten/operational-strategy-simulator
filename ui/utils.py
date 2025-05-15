import pandas as pd

def format_dataframe(df):
    df = df.copy()

    #df["Initiative"] = "<b>" + df["Initiative"] + "</b>"
    df["Cost ($)"] = df["Cost ($)"].apply(lambda x: f"${x:,.0f}")
    df["Engineering Days"] = df["Engineering Days"].apply(lambda x: f"{x:,}")
    df["Expected ARR Impact ($)"] = df["Expected ARR Impact ($)"].apply(lambda x: f"${x:,.0f}")
    df["ROI"] = df["ROI"].apply(lambda x: f"{x:.2f}")

    return df.style.set_table_attributes('class="styled-table"').set_properties(**{
        'text-align': 'left',
        'font-weight': 'bold'
    })
