import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set_theme()

def main():
    x_values = np.array([])
    y_values = np.array([])
    for day in range(1, 4):
        df_stock = pd.read_csv("Round4/implied_vol/coco_df_" + str(day) + ".csv", sep=',')
        x_values = np.concatenate((x_values, df_stock.mid_price))
        df_option = pd.read_csv("Round4/implied_vol/coup_df_" + str(day) + ".csv", sep=',')
        y_values = np.concatenate((y_values, df_option.imp_vol))

    q_values = (((x_values - 10000)/x_values)**2).reshape(-1, 1)
    q_train, q_test, y_train, y_test = train_test_split(q_values, y_values, test_size=0.2)


    plt.figure()
    plt.scatter(q_train, y_train, s=2, color='blue')
    plt.scatter(q_test, y_test, s=2, color='red')
    plt.show()

    results = np.zeros(shape=101)
    reg = LinearRegression().fit(q_train, y_train)
    print(reg.coef_[0], reg.intercept_)
    results[0] = reg.score(q_test, y_test)
    for i, a in enumerate(np.linspace(0.001, 0.1, 100)):
        reg = Lasso(alpha=a).fit(q_train, y_train)
        results[i+1] = reg.score(q_test, y_test)

    plt.figure()
    plt.plot(results)
    plt.show()

    print(np.argmax(results), results.max())

    return 0

if __name__ == "__main__":
    main()