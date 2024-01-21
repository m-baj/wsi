import glob
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def prepare_data():
    folder_path = "Diabetes-Data"
    csv_files = glob.glob(f"{folder_path}/data-*")

    combined_df = pd.DataFrame()
    for csv_file in csv_files:
        df = pd.read_csv(
            csv_file,
            delimiter="\t",
            header=None,
            names=["Date", "Time", "Code", "Value"],
        )
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    df = combined_df

    df.drop("Date", axis=1, inplace=True)

    if df["Time"].dtype == "O":  # 'O' oznacza object, co zwykle odpowiada stringom
        df["Hour"] = pd.to_datetime(df["Time"], errors="coerce", format="%H:%M").dt.hour
        df["Minute"] = pd.to_datetime(
            df["Time"], errors="coerce", format="%H:%M"
        ).dt.minute
    else:
        df["Hour"] = df["Time"].apply(lambda x: x.hour if x is not pd.NaT else None)
        df["Minute"] = df["Time"].apply(lambda x: x.minute if x is not pd.NaT else None)

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    scaler = StandardScaler()
    df["Value"] = scaler.fit_transform(df[["Value"]])
    df["Hour"] = scaler.fit_transform(df[["Hour"]])
    df["Minute"] = scaler.fit_transform(df[["Minute"]])

    df.drop("Time", axis=1, inplace=True)
    df = df.dropna(subset=["Value", "Hour", "Minute"])

    correct_codes = [
        33,
        34,
        35,
        48,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
    ]
    df = df[df["Code"].isin(correct_codes)]

    X = np.array(df.drop("Code", axis=1))

    onehot_encoder = OneHotEncoder(sparse=False)
    y = onehot_encoder.fit_transform(df[["Code"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
