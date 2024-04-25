import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

import seaborn as sns
sns.set_theme()


def main():
    day = 3
    df = pd.read_csv("Round4/round-4-island-data-bottle/prices_round_4_day_" + str(day) + ".csv", sep=';')
    df_stock = df[df["product"] == "COCONUT"]
    df_stock = df_stock.reset_index()
    df_option = df[df["product"] == "COCONUT_COUPON"]
    df_option = df_option.reset_index()
    df_stock["log_returns"] = np.log(df_stock.mid_price/df_stock.mid_price.shift(1))
    df_stock["prev_log_returns"] = df_stock.log_returns.shift(1)
    plt.figure()
    plt.scatter(df_stock.log_returns, df_stock.prev_log_returns)
    plt.show()

    print(df_stock[["prev_log_returns", "log_returns"]].corr())

    return 0

if __name__ == "__main__":
    main()