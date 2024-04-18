import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.linear_model import LinearRegression
from typing import OrderedDict

import seaborn as sns
sns.set_theme()

def analyse_data(number):
    results = np.zeros(shape=(3, 3))
    df = pd.read_csv("Round3/round-3-island-data-bottle/prices_round_3_day_" + str(number) + ".csv", sep=';')
    df_grouped = df.groupby(df["product"])
    df_chocolate = df_grouped.get_group("CHOCOLATE").reset_index()
    df_strawberries = df_grouped.get_group("STRAWBERRIES").reset_index()
    df_roses = df_grouped.get_group("ROSES").reset_index()
    df_gift_basket = df_grouped.get_group("GIFT_BASKET").reset_index()
    df_gift_basket["underlying_mid"] = 4*df_chocolate.mid_price + 6*df_strawberries.mid_price + df_roses.mid_price
    df_gift_basket["underlying_ask"] = 4*df_chocolate.ask_price_1 + 6*df_strawberries.ask_price_1 + df_roses.ask_price_1
    df_gift_basket["underlying_bid"] = 4*df_chocolate.bid_price_1 + 6*df_strawberries.bid_price_1 + df_roses.bid_price_1

    diff_mid = df_gift_basket.mid_price / df_gift_basket.underlying_mid
    diff_ask = df_gift_basket.ask_price_1 / df_gift_basket.underlying_ask
    diff_bid = df_gift_basket.bid_price_1 / df_gift_basket.underlying_bid

    results[0, 0] = diff_mid.mean()
    results[0, 1] = diff_mid.std()
    results[0, 2] = np.max(np.abs(diff_mid - results[0, 0]))

    results[1, 0] = diff_ask.mean()
    results[1, 1] = diff_ask.std()
    results[1, 2] = np.max(np.abs(diff_ask - results[1, 0]))

    results[2, 0] = diff_bid.mean()
    results[2, 1] = diff_bid.std()
    results[2, 2] = np.max(np.abs(diff_bid - results[2, 0]))

    return results

def pnl_calculator(df):
    pnl = np.zeros(shape=df.index.size)
    still_worth_buying = (df.ask_price_1 < df.fair_bid.shift(1))
    still_worth_buying[0] = False
    still_worth_selling = (df.bid_price_1 > df.fair_ask.shift(1))
    still_worth_selling[0] = False
    actual_trades = np.zeros(shape=df.index.size)
    actual_trades[still_worth_buying] = df.trades[still_worth_buying]
    actual_trades[still_worth_selling] = df.trades[still_worth_selling]
    mask_buy = (actual_trades > 0.5)
    mask_sell = (actual_trades < -0.5)
    # Buy product
    pnl[mask_buy] -= actual_trades[mask_buy] * df.ask_price_1.to_numpy()[mask_buy]
    # Sell product
    pnl[mask_sell] -= actual_trades[mask_sell] * df.bid_price_1.to_numpy()[mask_sell]
    pnl = np.cumsum(pnl)
    position = np.cumsum(actual_trades)
    mask_long = position > 0.5
    mask_short = position < -0.5
    # Close long position
    pnl[mask_long] += position[mask_long]*df.bid_price_1.to_numpy()[mask_long]
    # Close short position
    pnl[mask_short] += position[mask_short]*df.ask_price_1.to_numpy()[mask_short]

    return pnl

def compute_pnl_helper(df_gift_basket, good_sells, good_buys, pos_limit, step):
    aim_positions = np.zeros(shape=df_gift_basket.index.size)
    aim_positions[df_gift_basket.sell_opp_value > 0] = -np.floor_divide(good_sells, step)
    aim_positions[df_gift_basket.buy_opp_value > 0] = np.floor_divide(good_buys, step)

    aim_positions[aim_positions > pos_limit+0.5] = pos_limit
    aim_positions[aim_positions < -pos_limit - 0.5] = -pos_limit
    assert(np.max(np.abs(aim_positions)) < 60.5)

    trades = np.zeros(shape=df_gift_basket.index.size)
    trades = aim_positions - np.roll(aim_positions, 1)
    trades[0] = aim_positions[0]
    trades = np.roll(trades, 1)
    trades[0] = 0
    df_gift_basket["trades"] = trades

    pnl = pnl_calculator(df_gift_basket)

    return aim_positions, trades, pnl

def compute_pnl(number):
    df = pd.read_csv("Round3/round-3-island-data-bottle/prices_round_3_day_" + str(number) + ".csv", sep=';')
    df_grouped = df.groupby(df["product"])
    df_chocolate = df_grouped.get_group("CHOCOLATE").reset_index()
    df_strawberries = df_grouped.get_group("STRAWBERRIES").reset_index()
    df_roses = df_grouped.get_group("ROSES").reset_index()
    df_gift_basket = df_grouped.get_group("GIFT_BASKET").reset_index()
    df_gift_basket["underlying_mid"] = 4*df_chocolate.mid_price + 6*df_strawberries.mid_price + df_roses.mid_price
    df_gift_basket["underlying_ask"] = 4*df_chocolate.ask_price_1 + 6*df_strawberries.ask_price_1 + df_roses.ask_price_1
    df_gift_basket["underlying_bid"] = 4*df_chocolate.bid_price_1 + 6*df_strawberries.bid_price_1 + df_roses.bid_price_1
    df_gift_basket["fair_mid"] = df_gift_basket["underlying_mid"] * 1.0053
    df_gift_basket["fair_ask"] = df_gift_basket["underlying_ask"] * 1.0058
    df_gift_basket["fair_bid"] = df_gift_basket["underlying_bid"] * 1.0059

    df_gift_basket["sell_opp_value"] = df_gift_basket.bid_price_1 - df_gift_basket.fair_ask
    df_gift_basket["buy_opp_value"] = df_gift_basket.fair_bid - df_gift_basket.ask_price_1

    pos_limit = 60

    good_sells = df_gift_basket.sell_opp_value[df_gift_basket.sell_opp_value > 0].to_numpy()
    good_buys = df_gift_basket.buy_opp_value[df_gift_basket.buy_opp_value > 0].to_numpy()
    
    first = True
    max_pnl = 0
    max_pnl_step = 0
    lowest_pnl = 0
    low_risk_step = 0
    for step in np.linspace(0.01, 1, 100):
        _, _, pnl = compute_pnl_helper(df_gift_basket, good_sells, good_buys, pos_limit, step)
        final_pnl = pnl[-1]
        current_lowest_pnl = pnl.min()
        if (final_pnl > max_pnl) or first:
            max_pnl = final_pnl
            max_pnl_step = step
        if (current_lowest_pnl > lowest_pnl) or first:
            current_lowest_pnl = lowest_pnl
            low_risk_step = step
        first = False

    low_risk_step = 1
    _, _, pnl_max = compute_pnl_helper(df_gift_basket, good_sells, good_buys, pos_limit, max_pnl_step)
    max_positions = np.cumsum(df_gift_basket.trades.to_numpy())
    _, _, pnl_min = compute_pnl_helper(df_gift_basket, good_sells, good_buys, pos_limit, low_risk_step)
    min_positions = np.cumsum(df_gift_basket.trades.to_numpy())

    plt.figure()
    plt.plot(pnl_max, color="red", label="Step:" + str(max_pnl_step))
    plt.plot(pnl_min, color="green", label="Step:" + str(low_risk_step))
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(max_positions, color="red", label="Step:" + str(max_pnl_step))
    plt.plot(min_positions, color="green", label="Step:" + str(low_risk_step))
    plt.legend()
    plt.show()

    return max_pnl, max_pnl_step, lowest_pnl, low_risk_step

def main():
    data_analysis = False
    pnl = True

    if data_analysis:
        results = np.zeros(shape=(4, 3, 3))
        for number in range(4):
            results[number, :] = analyse_data(number)
        print(results)
        print(results.mean(axis=0))

    if pnl:
        for number in range(4):
            res = compute_pnl(number)
            print("Dataset:", number)
            print("Max PnL:", res[0])
            print("Max PnL step:", res[1])
            print("Low risk PnL:", res[2])
            print("Low risk step:", res[3])
    
    return 0



if __name__ == "__main__":
    main()
