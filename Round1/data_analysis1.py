import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set_theme()

def errors(df, a):
    df['log_returns_prev'] = df.log_returns.shift(1).ewm(alpha=a).mean()
    df['log_ask_returns_prev'] = df.log_ask_returns.shift(1).ewm(alpha=a).mean()
    df['log_bid_returns_prev'] = df.log_bid_returns.shift(1).ewm(alpha=a).mean()
    X, y = df.log_returns_prev.to_numpy().reshape(-1, 1)[5:], df.log_returns.to_numpy()[5:]
    X_ask, y_ask = df.log_ask_returns_prev.to_numpy().reshape(-1, 1)[5:], df.log_ask_returns.to_numpy()[5:]
    X_bid, y_bid = df.log_bid_returns_prev.to_numpy().reshape(-1, 1)[5:], df.log_bid_returns.to_numpy()[5:]
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    reg_ask = LinearRegression(fit_intercept=False).fit(X_ask, y_ask)
    reg_bid = LinearRegression(fit_intercept=False).fit(X_bid, y_bid)
    r = reg.score(X, y)
    r_ask = reg_ask.score(X_ask, y_ask)
    r_bid = reg_bid.score(X_bid, y_bid)
    return np.array([r, r_ask, r_bid])

def load_database(number):
    df = pd.read_csv("Round1/round-1-island-data-bottle/prices_round_1_day_" + str(number) + ".csv", sep=';')
    dfs = df.loc[df["product"] == "STARFRUIT"]
    dfs['log_returns'] = np.log(dfs.mid_price/dfs.mid_price.shift(1))
    dfs['log_ask_returns'] = np.log(dfs.ask_price_1/dfs.ask_price_1.shift(1))
    dfs['log_bid_returns'] = np.log(dfs.bid_price_1/dfs.bid_price_1.shift(1))

    return dfs

def main():
    p = 100

    rs = np.zeros(shape=(3, p, 3))

    for number in range(-2, 1):
        df = load_database(number)
        for i, a in enumerate(np.linspace(0.01, 1, p)):
            rs[number+2, i, :] = errors(df, a)

    fmt = lambda x: "{:.2f}".format(x)

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 8))
    fig.suptitle('Starfruit')

    ax1.plot(rs[0, :, 0], label = "-2: mid")
    ax1.plot(rs[0, :, 1], label = "-2: ask")
    ax1.plot(rs[0, :, 2], label = "-2: bid")
    ax1.set(xlabel='Alpha', ylabel='R squared')
    ax1.set_xticks(range(0, 101, 10), [fmt(t) for t in np.linspace(0, 1, 11)])
    ax1.legend()

    ax2.plot(rs[1, :, 0], label = "-1: mid")
    ax2.plot(rs[1, :, 1], label = "-1: ask")
    ax2.plot(rs[1, :, 2], label = "-1: bid")
    ax2.set(xlabel='Alpha', ylabel='R squared')
    ax2.set_xticks(range(0, 101, 10), [fmt(t) for t in np.linspace(0, 1, 11)])
    ax2.legend()

    ax3.plot(rs[2, :, 0], label = "0: mid")
    ax3.plot(rs[2, :, 1], label = "0: ask")
    ax3.plot(rs[2, :, 2], label = "0: bid")
    ax3.set(xlabel='Alpha', ylabel='R squared')
    ax3.set_xticks(range(0, 101, 10), [fmt(t) for t in np.linspace(0, 1, 11)])
    ax3.legend()

    results = rs.mean(axis=0)

    print("Best alpha mid", np.argmax(results[:, 0]) / p)
    print("Best alpha ask", np.argmax(results[:, 1]) / p)
    print("Best alpha bid", np.argmax(results[:, 2]) / p)
    print("R squared mid", results[:, 0].max())
    print("R squared ask", results[:, 1].max())
    print("R squared bid", results[:, 2].max())




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
