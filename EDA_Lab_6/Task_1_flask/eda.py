import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # one-hot
    df = pd.get_dummies(
        df,
        columns=["Month", "VisitorType"],
        drop_first=True
    )

    # label encoding
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[col] = df[col].astype("category").cat.codes

    # bool → int
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_yes = df[df["Revenue"] == 1]
    df_no = df[df["Revenue"] == 0]

    df_no_down = df_no.sample(
        n=len(df_yes),
        random_state=42
    )

    df_balanced = pd.concat([df_yes, df_no_down])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("online_shoppers_intention.csv")

    df_processed = preprocess(df)
    df_balanced = balance_dataset(df_processed)

    df_balanced.to_csv("prepared_data.csv", index=False)

    print("EDA completed. File saved: prepared_data.csv")
