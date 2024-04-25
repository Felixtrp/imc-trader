import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import seaborn as sns
sns.set_theme()

def visualise(dfa, trader, product, day):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs_twins = np.array([a.twinx() for a in axs.flat]).reshape(2, 3)

    fig.suptitle(trader + " trading " + product + " on day " + str(day))

    axs[0,0].set_title("Total PnL")
    axs[0,0].plot(dfa.pnl, label="PnL", color="red", linewidth=0.4)
    axs_twins[0,0].plot(dfa.mid_price, alpha=0.2, label="Mid price", color="blue", linewidth=0.4)
    axs[0,0].legend()

    axs[0,1].set_title("Market making PnL")
    axs[0,1].plot(dfa.mm_pnl, label="PnL", color="green", linewidth=0.4)
    axs_twins[0,1].plot(dfa.mid_price, alpha=0.2, label="Mid price", color="blue", linewidth=0.4)
    axs[0,1].legend()

    axs[0,2].set_title("Market taking PnL")
    axs[0,2].plot(dfa.mt_pnl, label="PnL", color="orange", linewidth=0.4)
    axs_twins[0,2].plot(dfa.mid_price, alpha=0.2, label="Mid price", color="blue", linewidth=0.4)
    axs[0,2].legend()

    axs[1,0].set_title("Total Position")
    axs[1,0].plot(dfa.position, label="Position", color="red", linewidth=0.8)
    axs_twins[1,0].plot(dfa.mid_price, alpha=0.2, label="Mid price", color="blue", linewidth=0.4)
    axs[1,0].legend()

    axs[1,1].set_title("Market making Position")
    axs[1,1].plot(dfa.mm_position, label="Position", color="green", linewidth=0.8)
    axs_twins[1,1].plot(dfa.mid_price, alpha=0.2, label="Mid price", color="blue", linewidth=0.4)
    axs[1,1].legend()

    axs[1,2].set_title("Market taking Position")
    axs[1,2].plot(dfa.mt_position, label="Position", color="orange", linewidth=0.8)
    axs_twins[1,2].plot(dfa.mid_price, alpha=0.2, label="Mid price", color="blue", linewidth=0.4)
    axs[1,2].legend()

    for ax in axs[0, :]:
        ax.set(xlabel='Timestamp', ylabel='PnL')
    for ax in axs[1, :]:
        ax.set(xlabel='Timestamp', ylabel='Position')
    
    for ax in axs.flat:
        ax.label_outer()
    
    for ax in axs_twins.flat:
        ax.label_outer()

    plt.show()

    return None

def data_analysis(trader: str, product: str, round: int, day: int, plot=False):
    df = pd.read_csv("Round5/round-5-island-data-bottle/trades_round_" + str(round) + "_day_" + str(day) + "_wn.csv", sep=';')
    mid_prices = pd.read_csv("Round" + str(round) + "/round-" + str(round) + "-island-data-bottle/prices_round_" + str(round) + "_day_" + str(day) + ".csv", sep=';')
    mid_prices = mid_prices[mid_prices['product'] == product]
    if mid_prices.empty:
        raise AssertionError(product + " is not traded on day " + str(day))
    mid_prices.set_index('timestamp', inplace=True)

    # # Double checking
    # df = df.iloc[:10]
    # last = df.timestamp.iloc[-1]
    # close = mid_prices.loc[last].mid_price
    # print(df)
    # print(mid_prices.mid_price.loc[df.timestamp])
    # df_buy = df[(df.buyer == trader) & (df.symbol == product)]
    # df_sell = df[(df.seller == trader) & (df.symbol == product)]
    # revenue = (df_sell.price * df_sell.quantity).sum() - (df_buy.price * df_buy.quantity).sum()
    # position = np.sum(df_buy.quantity) - np.sum(df_sell.quantity)
    # return revenue + position*close
    
    df_buy = df[(df.buyer == trader) & (df.seller != trader) & (df.symbol == product)]
    df_buy['total_price'] = df_buy.price * df_buy.quantity
    df_buy = df_buy[['timestamp', 'quantity', 'total_price']].groupby("timestamp").sum()

    df_sell = df[(df.buyer != trader) & (df.seller == trader) & (df.symbol == product)]
    df_sell['total_price'] = df_sell.price * df_sell.quantity
    df_sell = df_sell[['timestamp', 'quantity', 'total_price']].groupby("timestamp").sum()

    dfa = pd.DataFrame(columns=['mid_price',
                                'buy_volume',
                                'buy_price',
                                'sell_volume',
                                'sell_price',
                                'trades',
                                'market_maker_buy',
                                'market_maker_sell',
                                'mm_buy_volume',
                                'mm_sell_volume',
                                'mm_trades',
                                'mt_buy_volume',
                                'mt_sell_volume',
                                'mt_trades',
                                'mm_position',
                                'mt_position',
                                'position',
                                'mm_pnl',
                                'mt_pnl',
                                'pnl'
                                ],
                       index=np.arange(0, 1000000, 100)
                       )
    dfa.index.name = "timestamp"

    dfa['mid_price'] = mid_prices.mid_price
    dfa['buy_volume'] = df_buy.quantity
    dfa['buy_price'] = df_buy.total_price / df_buy.quantity
    dfa['sell_volume'] = df_sell.quantity
    dfa['sell_price'] = df_sell.total_price / df_sell.quantity
    dfa[['buy_volume', 'sell_volume']] = dfa[['buy_volume', 'sell_volume']].fillna(value=0)
    dfa['trades'] = dfa.buy_volume - dfa.sell_volume

    dfa['market_maker_buy'][dfa.buy_volume > 0] = (dfa.buy_price < dfa.mid_price)[dfa.buy_volume > 0]
    dfa['market_maker_sell'][dfa.sell_volume > 0] = (dfa.sell_price > dfa.mid_price)[dfa.sell_volume > 0]

    dfa['mm_buy_volume'] = dfa.buy_volume * dfa.market_maker_buy.fillna(False)
    dfa['mm_sell_volume'] = dfa.sell_volume * dfa.market_maker_sell.fillna(False)
    dfa['mm_trades'] = dfa.mm_buy_volume - dfa.mm_sell_volume
    dfa['mt_buy_volume'] = dfa.buy_volume * (~dfa.market_maker_buy.fillna(True))
    dfa['mt_sell_volume'] = dfa.sell_volume * (~dfa.market_maker_sell.fillna(True))
    dfa['mt_trades'] = dfa.mt_buy_volume - dfa.mt_sell_volume

    dfa['mm_position'] = (dfa.mm_buy_volume - dfa.mm_sell_volume).cumsum()
    dfa['mt_position'] = (dfa.mt_buy_volume - dfa.mt_sell_volume).cumsum()
    dfa['position'] = (dfa.buy_volume - dfa.sell_volume).cumsum()

    mm_revenue = dfa.mm_sell_volume * dfa.sell_price.fillna(0) - dfa.mm_buy_volume * dfa.buy_price.fillna(0)
    dfa['mm_pnl'] = mm_revenue.cumsum() + dfa.mm_position * dfa.mid_price
    mt_revenue = dfa.mt_sell_volume * dfa.sell_price.fillna(0) - dfa.mt_buy_volume * dfa.buy_price.fillna(0)
    dfa['mt_pnl'] = mt_revenue.cumsum() + dfa.mt_position * dfa.mid_price
    revenue = dfa.sell_volume * dfa.sell_price.fillna(0) - dfa.buy_volume * dfa.buy_price.fillna(0)
    dfa['pnl'] = revenue.cumsum() + dfa.position * dfa.mid_price

    if plot:
        visualise(dfa, trader, product, day)

    return dfa

def find_traders(symbol, round, day):
    df = pd.read_csv("Round5/round-5-island-data-bottle/trades_round_" + str(round) + "_day_" + str(day) + "_wn.csv", sep=';')
    df_symbol = df[df.symbol == symbol]
    res = np.concatenate((df_symbol.buyer.to_numpy(), df_symbol.seller.to_numpy()))
    return np.unique(res)

def print_dataframe():
    df = pd.DataFrame(columns=['trader',
                               'symbol',
                               'round',
                               'day',
                               "total_pnl",
                               "mm_pnl",
                               "mt_pnl",
                               "total_buy_volume",
                               "total_sell_volume",
                               "mm_buy_volume",
                               "mm_sell_volume",
                               "mt_buy_volume",
                               "mt_sell_volume",
                               ])
    counter = 0
    for symbol in ["AMETHYSTS", "STARFRUIT", "GIFT_BASKET", "ROSES", "CHOCOLATE", "STRAWBERRIES", "COCONUT", "COCONUT_COUPON"]:
        if symbol == "AMETHYSTS" or symbol == "STARFRUIT":
            round = 1
            days = [-2, -1, 0]
        elif symbol == "GIFT_BASKET" or symbol == "ROSES" or symbol == "CHOCOLATE" or symbol == "STRAWBERRIES":
            round = 3
            days = [0, 1, 2]
        else:
            round = 4
            days = [1, 2, 3]
        
        for day in days:
            all_traders = find_traders(symbol, round, day)

            for trader in all_traders:
                dfa = data_analysis(trader, symbol, round, day, plot=False)
                df.loc[counter] = [trader,
                                   symbol,
                                   round,
                                   day,
                                   dfa.pnl.iloc[-1],
                                   dfa.mm_pnl.iloc[-1],
                                   dfa.mt_pnl.iloc[-1],
                                   dfa.buy_volume.sum(),
                                   -dfa.sell_volume.sum(),
                                   dfa.mm_buy_volume.sum(),
                                   -dfa.mm_sell_volume.sum(),
                                   dfa.mt_buy_volume.sum(),
                                   -dfa.mt_sell_volume.sum(),
                                   ]
                counter += 1

    df.to_csv('all_traders.csv', index=False)
    return None

def main():
    df = pd.read_csv("Round5/all_traders.csv")
    # for symbol in ["AMETHYSTS", "STARFRUIT", "GIFT_BASKET", "ROSES", "CHOCOLATE", "STRAWBERRIES", "COCONUT", "COCONUT_COUPON"]:
    #     dg = df[df.symbol == symbol]
    #     print(symbol)
    #     print(dg[["trader", "total_pnl", "mm_pnl", "mt_pnl", "total_buy_volume", "total_sell_volume",]].groupby("trader").sum())
    #     print('\n')
    
    print(df[(df.symbol == "COCONUT_COUPON")][['trader', 'mm_pnl', 'mt_pnl']])

    # trader = "Rhianna"
    symbol = "COCONUT_COUPON"
    round = 4
    days = list(range(round-3, round))

    # dfa1 = data_analysis(trader, symbol, round, days[0], plot=False)
    # dfa2 = data_analysis(trader, symbol, round, days[1], plot=False)
    # # dfa2.index += 1000000
    # dfa3 = data_analysis(trader, symbol, round, days[2], plot=False)
    # # dfa3.index += 2000000
    # # dfa = pd.concat([dfa1, dfa2, dfa3])
    # dfa = [dfa1, dfa2, dfa3]

    # current = dfa1
    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax_twin = ax.twinx()
    # ax.plot(current.mid_price)
    # ax_twin.plot(current.mt_trades)
    # plt.show()

    traders = find_traders(symbol, round, days[0])

    lims = range(0, 16, 1)
    results = {} # np.zeros(shape=(len(traders), 3, len(lims)))
    for trader in traders:
        partial = np.zeros(shape=(3, len(lims)))
        for j, day in enumerate(days):
            dfa = data_analysis(trader, symbol, round, day, plot=False)
            for k, lim in enumerate(lims):
                mask_buy = (dfa.mt_trades >= lim).shift(0).fillna(False)
                mask_sell = (dfa.mt_trades <= -lim).shift(0).fillna(False)
                rough_pnl = -(dfa.mid_price[mask_buy]).sum()  + (dfa.mid_price[mask_sell]).sum()
                rough_pnl += (np.sum(mask_buy) - np.sum(mask_sell)) * dfa.mid_price.iloc[-1]
                partial[j, k] = rough_pnl

        results[trader] = partial

    for trader in traders:

        means = results[trader].mean(axis=0)
        ind_mean = np.argmax(means)
        print("Trader:", trader)
        print("Best avg threshold:", lims[ind_mean])
        print("Profit:", results[trader][:, ind_mean])

    # print(results["Vladimir"])
    plot = True
    if plot:
        dfa = pd.DataFrame()
        for i in range(3):
            next_dfa = data_analysis("Vinnie", symbol, round, days[i], plot=True)
            next_dfa.index += i*1000000
            dfa = pd.concat([dfa, next_dfa])
        # _, ax = plt.subplots(figsize=(12, 8))
        # ax_twin = ax.twinx()
        # ax.plot(dfa.mid_price)
        # ax_twin.plot(dfa.mt_trades, color="orange", linewidth=0.3)
        # plt.show()

    return 0

if __name__ == "__main__":
    main()