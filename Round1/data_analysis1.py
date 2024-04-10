import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import seaborn as sns
sns.set_theme()

def profit(number, a):
    df = pd.read_csv("round-1-island-data-bottle/prices_round_1_day_" + str(-number) + ".csv", sep=';')
    dfs = df.loc[df["product"] == "STARFRUIT"]
    dfs['log_returns'] = np.log(dfs.mid_price) - np.log(dfs.mid_price.shift(1))
    dfs['log_returns_prev'] = dfs['log_returns'].shift(1).ewm(alpha=a).mean()
    r = dfs[['log_returns', 'log_returns_prev']].corr().to_numpy()[0, 1]
    coeff = (r * dfs.log_returns.std()) / dfs.log_returns_prev.std()
    dfs["predicted_log_returns"] = coeff*dfs.log_returns_prev
    dfs['random_element'] = dfs.log_returns - dfs.predicted_log_returns
    std = dfs.random_element.std()

    buy_pnl = (dfs.predicted_log_returns.to_numpy()[2:] > 0) * (dfs.mid_price - dfs.mid_price.shift(1)).to_numpy()[2:]
    sell_pnl = (dfs.predicted_log_returns.to_numpy()[2:] < 0) * (dfs.mid_price.shift(1) - dfs.mid_price).to_numpy()[2:]
    pnl = np.sum(buy_pnl + sell_pnl)

    return pnl, std


def main():
    p = 100

    pnls = np.zeros(shape=(3, p))
    stds = np.zeros(shape=(3, p))

    for number in range(3):
        for i, a in enumerate(np.linspace(0.01, 1, p)):
            pnls[number, i], stds[number, i] = profit(number, a)

    fmt = lambda x: "{:.2f}".format(x)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 8))
    fig.suptitle('Starfruit')

    ax1.plot(pnls[0, :], label = "0")
    ax1.plot(pnls[1, :], label = "-1")
    ax1.plot(pnls[2, :], label = "-2")
    ax1.set(xlabel='Alpha', ylabel='PnL')
    ax1.set_xticks(range(0, 101, 10), [fmt(t) for t in np.linspace(0, 1, 11)])
    ax1.legend()

    ax2.plot(stds[0,:], label = "0")
    ax2.plot(stds[1,:], label = "-1")
    ax2.plot(stds[2,:], label = "-2")
    ax2.set(xlabel='Alpha', ylabel='Risk')
    ax2.set_xticks(range(0, 101, 10), [fmt(t) for t in np.linspace(0, 1, 11)])
    ax2.legend()

    plt.show()



    #my_normal = np.random.normal(loc=0, scale=std, size=dfs.random_element.size)

    #lims = np.array([-0.00115, 0.00115])

    #print(mean, std)

    #plt.hist([dfs.random_element, my_normal], bins=np.linspace(-0.00115, 0.00115, 24), label=['nat', 'art'])
    #plt.show()

    #plt.figure()
    #plt.scatter(dfs.log_returns_prev, dfs.log_returns, marker='.')
    #plt.plot(lims, coeff*lims)
    #plt.show()

    return 0

if __name__ == "__main__":
    main()