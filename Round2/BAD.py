from typing import List, Any, Dict
import string
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import defaultdict, OrderedDict
import copy
import numpy as np
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()


class Trader:
    def __init__(self) -> None:
        self.prev_state = 0
    def run(self, state: TradingState):
        conversions = 0
        result = {}
        trader_data = ''
        straw_mid = self.get_mid_price(state.order_depths['STRAWBERRIES'])
        choco_mid = self.get_mid_price(state.order_depths['CHOCOLATE'])
        rose_mid = self.get_mid_price(state.order_depths['ROSES'])
        gift_mid = self.get_mid_price(state.order_depths['GIFT_BASKET'])

        signal = straw_mid * 6 + choco_mid * 4 + rose_mid - gift_mid + 380
        std = 76.42
        BUY_LEVELS = {1:[-2*std, (-60, -60)], 
                      2:[-std, (-60, -50)], 
                      3:[-30, (-50, -20)], 
                      4:[-1, (-20, 0)], 
                      5:[0, (0, 0)],
                      6:[0, (0, 0)],
                      7:[1, (0, 20)], 
                      8:[30, (20, 50)], 
                      9:[std, (50, 60)], 
                      10:[2*std, (60, 60)]}
        if signal < BUY_LEVELS[1][0]:
            curr_level = 1
        elif BUY_LEVELS[1][0] <= signal < BUY_LEVELS[2][0]:
            curr_level = 2
        elif BUY_LEVELS[2][0] <= signal < BUY_LEVELS[3][0]:
            curr_level = 3
        elif BUY_LEVELS[3][0] <= signal < BUY_LEVELS[4][0]:
            curr_level = 4
        elif BUY_LEVELS[4][0] <= signal < BUY_LEVELS[5][0]:
            curr_level = 5
        elif BUY_LEVELS[5][0] <= signal < BUY_LEVELS[6][0]:
            curr_level = 6
        elif BUY_LEVELS[6][0] <= signal < BUY_LEVELS[7][0]:
            curr_level = 7
        elif BUY_LEVELS[7][0] <= signal < BUY_LEVELS[8][0]:
            curr_level = 8
        elif signal >= BUY_LEVELS[8][0]:
            curr_level = 10
        #elif signal >= BUY_LEVELS[10][0]: #this somehow won't work with the backtester
        #    curr_level = 10
        #else:
        #    for i in range(2, 10):
        #        if BUY_LEVELS[i - 1][0] <= signal < BUY_LEVELS[i][0]:
        #            curr_level = i
        #        else:
        #            logger.print(f"Signal {signal} does not fall into range {BUY_LEVELS[i - 1][0]} to {BUY_LEVELS[i][0]}")

        for product in state.order_depths:
            orders: list[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            osell = OrderedDict(sorted(order_depth.sell_orders.items()))
            obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
            sell_vol, best_sell_pr = self.values_extract(osell) #4000
            buy_vol, best_buy_pr = self.values_extract(obuy, 1) #3999
            curr_pos = 0
            if product == 'STRAWBERRIES':
                buy_pr = max(order_depth.buy_orders.keys())
                sell_pr = min(order_depth.sell_orders.keys())
                LIMIT = 348
                if product in state.position.keys():
                    curr_pos = state.position[product]
                low, high = BUY_LEVELS[curr_level][1]
                high, low = -math.floor(low/60 * LIMIT), -math.floor(high/60 * LIMIT)
                logger.print(curr_pos, low, high)
                if high > 0:
                    if curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos))
                        logger.print('a', sell_pr)
                    elif curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                        logger.print('b', buy_pr)
                elif high <= 0:
                    if curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                        logger.print('c', sell_pr)
                    elif curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos))
                        logger.print('d', buy_pr)
                result[product] = orders
            if product == 'CHOCOLATE':
                buy_pr = max(order_depth.buy_orders.keys())
                sell_pr = min(order_depth.sell_orders.keys())
                LIMIT = 232
                if product in state.position.keys():
                    curr_pos = state.position[product]
                low, high = BUY_LEVELS[curr_level][1]
                high, low = -math.floor(low/60 * LIMIT), -math.floor(high/60 * LIMIT)
                if high > 0:
                    if curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos))
                    elif curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                elif high <= 0:
                    if curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                    elif curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos))
                result[product] = orders
            if product == 'ROSES':
                buy_pr = max(order_depth.buy_orders.keys())
                sell_pr = min(order_depth.sell_orders.keys())
                LIMIT = 58
                if product in state.position.keys():
                    curr_pos = state.position[product]
                low, high = BUY_LEVELS[curr_level][1]
                high, low = -low, -high
                if high > 0:
                    if curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos))
                    elif curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                elif high <= 0:
                    if curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                    elif curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos))
                result[product] = orders
            if product == 'GIFT_BASKET':
                buy_pr = max(order_depth.buy_orders.keys())
                sell_pr = min(order_depth.sell_orders.keys())
                LIMIT = 58
                if product in state.position.keys():
                    curr_pos = state.position[product]
                low, high = BUY_LEVELS[curr_level][1]
                if high > 0: #want to go up, buy
                    if curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos))
                    elif curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                elif high <= 0:
                    if curr_pos < low:
                        orders.append(Order(product, sell_pr, low - curr_pos))
                    elif curr_pos > high:
                        orders.append(Order(product, buy_pr, high - curr_pos)) # gift_mid + 1
                result[product] = orders
        logger.print(signal)
        logger.print(best_buy_pr, best_sell_pr)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if buy==0:
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def market_taker_buy(self, product, orders, osell, curr_pos, acc_bid, LIMIT, did_buy):
        for ask, vol in osell.items():
            if ask <= acc_bid:
                ask_vol = min(-vol, LIMIT - curr_pos - did_buy)
                if ask_vol > 0:
                    orders.append(Order(product, ask, ask_vol)) # postitive volume
                    did_buy += ask_vol
        return did_buy, orders

    def market_taker_sell(self, product, orders, obuy, curr_pos, acc_ask, LIMIT, did_sell):
        for bid, vol in obuy.items():
            if bid >= acc_ask:
                bid_vol = max(-vol, -LIMIT - curr_pos - did_sell)
                if bid_vol < 0:
                    orders.append(Order(product, bid, bid_vol)) # negative volume
                    did_sell += bid_vol
        return did_sell, orders

    def pair_trading1(self, product, order, obuy, osell, curr_pos, price_mid, LIMIT, signal, val):
        if signal > val:
            _, order = self.market_taker_sell(product, order, obuy, curr_pos, price_mid, LIMIT, 0)
        if signal < 1 and curr_pos < 0:
            _, order = self.market_taker_buy(product, order, osell, curr_pos, price_mid, LIMIT, 0)
        if signal < -val:
            _, order = self.market_taker_buy(product, order, osell, curr_pos, price_mid, LIMIT, 0)
        if signal > -1 and curr_pos > 0:
            _, order = self.market_taker_sell(product, order, obuy, curr_pos, price_mid, LIMIT, 0)
        return order

    def pair_trading2(self, product, order, obuy, osell, curr_pos, price_mid, LIMIT, signal, val, amt): # gift basket only
        if signal < -val:
            order.append(Order(product, int(price_mid) - amt, -LIMIT - curr_pos))
            #_, order = self.market_taker_sell(product, order, obuy, curr_pos, price_mid, LIMIT, 0)
        if signal > -1 and curr_pos < 0:
            order.append(Order(product, int(price_mid) + amt, -curr_pos))
            #_, order = self.market_taker_buy(product, order, osell, curr_pos, price_mid, LIMIT, 0)
        if signal > val:
            order.append(Order(product, int(price_mid) + amt, LIMIT - curr_pos))
            #_, order = self.market_taker_buy(product, order, osell, curr_pos, price_mid, LIMIT, 0)
        if signal < 1 and curr_pos > 0:
            order.append(Order(product, int(price_mid) - amt, -curr_pos))
            #_, order = self.market_taker_sell(product, order, obuy, curr_pos, price_mid, LIMIT, 0)
        return order
    
    def get_mid_price(self, depth):
        return (max(depth.sell_orders.keys()) + min(depth.buy_orders.keys())) / 2
