import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

import seaborn as sns
sns.set_theme()

def main():
    day = 2
    df_stock = pd.read_csv("Round4/implied_vol/coco_df_" + str(day) + ".csv", sep=',')
    df_option = pd.read_csv("Round4/implied_vol/coup_df_" + str(day) + ".csv", sep=',')

    print(df_stock)
    print(df_option)

    return 0

if __name__ == "__main__":
    main()