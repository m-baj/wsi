import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLUMNS = [
    "parametr kroku",
    "punkt początkowy",
    "wynik działania algorytmu",
    "czas trwania",
    "liczba iteracji",
]
SORT_BY = "parametr kroku"


def generate_table(data, function, expected_global_min):
    df = pd.DataFrame(data, columns=COLUMNS)
    df_sorted = df.sort_values(by=SORT_BY)
    df_sorted["czy znaleziono minimum globalne"] = df_sorted[
        "wynik działania algorytmu"
    ].apply(lambda x: np.isclose(function.f(x), expected_global_min, rtol=1e-3))
    df_sorted["czy znaleziono minimum globalne"] = df_sorted[
        "czy znaleziono minimum globalne"
    ].apply(lambda x: "Tak" if x else "Nie")
    return df_sorted


def plot_data(df):
    grouped = get_succsess_rate_for_all_learning_rates(df)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    grouped.plot(
        kind="bar",
        title="Liczba eksperymentów, w których znaleziono minimum globalne",
    )
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=45, ha="right")
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.show()


def get_succsess_rate_for_all_learning_rates(df):
    return df.groupby("parametr kroku")[
        "czy znaleziono minimum globalne"
    ].value_counts()
