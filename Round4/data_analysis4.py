import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

import seaborn as sns
sns.set_theme()

def black_scholes(day, strike=10000, rate=0, vol=0.0001):
    trading_days = 250
    
    df = pd.read_csv("Round4/round-4-island-data-bottle/prices_round_4_day_" + str(day) + ".csv", sep=';')
    df_stock = df[df["product"] == "COCONUT"]
    df_stock = df_stock.reset_index()
    df_option = df[df["product"] == "COCONUT_COUPON"]
    df_option = df_option.reset_index()

    time_to_expiry_beginning = 1 - (day-1)/trading_days
    # annualised_vols = (trading_days * df_option.index.size)**(1/2) * np.array([0.00010334488893431566, 0.00010246080046781773, 0.00010300342694559128])
    annualised_vol = vol*(trading_days * df_option.index.size)**(1/2)

    df_stock["log_returns"] = np.log(df_stock.mid_price/df_stock.mid_price.shift(1))

    df_option["time_to_expiry"] = time_to_expiry_beginning - (df_option.index / (250*df_option.index.size)) #np.ones(shape=df_option.index.size) # 
    df_option["forward_value"] = strike*np.exp(-rate*df_option.time_to_expiry)
    df_option["vol_adjust"] = annualised_vol * np.sqrt(df_option.time_to_expiry)#df_stock.volatility * np.sqrt(df_option.time_to_expiry)
    df_option["d_plus"] = (1/df_option.vol_adjust) * (np.log(df_stock.mid_price / strike) + (rate + annualised_vol*annualised_vol/2)*df_option.time_to_expiry)
    df_option["d_minus"] = df_option.d_plus - df_option.vol_adjust
    df_option["fair_value"] = norm.cdf(df_option.d_plus)*df_stock.mid_price - norm.cdf(df_option.d_minus)*df_option.forward_value

    df_option = df_option.drop(columns=["vol_adjust", "d_plus", "d_minus"])
    return df_stock, df_option



def main():
    strike = 10000
    rate = 0
    day = 1
    vol = 0.000101
    stock, option = black_scholes(day, strike, rate, vol) 
    diff = option.mid_price - option.fair_value

    # low = min(option.fair_value.min(), option.mid_price.min())
    # high = max(option.fair_value.max(), option.mid_price.max())
    # plt.figure()
    # plt.scatter(option.fair_value, option.mid_price, marker = '.')
    # plt.plot([low, high], [low, high])
    # plt.show()

    plt.figure()
    plt.plot(diff, label="Difference")
    plt.legend()
    plt.show()

    print(diff.mean(), diff.std())

    # plt.figure()
    # plt.plot(df_coconut.volatility.to_numpy())
    # plt.hlines(avg_volatility, 0, 10000)
    # plt.show()

    print(stock.log_returns.std())
    plt.figure()
    plt.hist([stock.log_returns.to_numpy(),
              np.random.normal(loc=0, scale=0.0000925, size=(stock.index.size-1))],
              label=["Log returns", "Normal"],
              bins=np.linspace(-0.00045, 0.00045, 20))
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()