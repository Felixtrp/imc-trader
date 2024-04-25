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
        self.t = 0
        self.event = 0
    def run(self, state: TradingState):
        self.t = state.timestamp
        conversions = 0
        result = {}
        trader_data = ''
        rose_mid = self.get_mid_price(state.order_depths['ROSES'])

        if 'ROSES' in state.market_trades.keys():
            curr = state.market_trades['ROSES']
            for trade in curr:
                if trade.timestamp == self.t - 100:
                    if trade.buyer == 'Rhianna':
                        self.event = 1
                    elif trade.seller == 'Rhianna':
                        self.event = -1
        for product in ['GIFT_BASKET', 'STRAWBERRIES', 'CHOCOLATE', 'ROSES']:
            orders: list[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            curr_pos = 0

            if product == 'ROSES':
                buy_pr = next(iter(order_depth.buy_orders.keys()), rose_mid - 1)
                sell_pr = next(iter(order_depth.sell_orders.keys()), rose_mid + 1)
                LIMIT = 60
                if product in state.position.keys():
                    curr_pos = state.position[product]
                if self.event == -1:
                    orders.append(Order(product, buy_pr, -LIMIT - curr_pos))
                elif self.event == 1:
                    orders.append(Order(product, sell_pr, LIMIT - curr_pos))
                result[product] = orders
                
        return result, conversions, trader_data

    
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
    
    def pair_trading1(self, product, order, best_buy, best_sell, curr_pos, LIMIT, signal, val):
        if signal > val:
            order.append(Order(product, best_buy, -LIMIT - curr_pos))
        if signal < 1 * 10 **-1 and curr_pos < 0:
            order.append(Order(product, best_sell, -curr_pos))
        if signal < -val:
            order.append(Order(product, best_sell, LIMIT - curr_pos))
        if signal > -1* 10 **-1 and curr_pos > 0:
            order.append(Order(product, best_buy, -curr_pos))
        return order
    
    def stanford(self, product, order, best_buy, best_sell, curr_pos, LIMIT, signal, val):
        if signal > val:
            order.append(Order(product, best_buy, -LIMIT - curr_pos))
        if signal < -val:
            order.append(Order(product, best_sell, LIMIT - curr_pos))
        return order
    
    def pair_trading2(self, product, order, mid, m, curr_pos, LIMIT, signal, val):
        if signal > val:
            order.append(Order(product, mid - m, -LIMIT - curr_pos))
        if signal < 1 and curr_pos < 0:
            order.append(Order(product, mid + m, -curr_pos))
        if signal < -val:
            order.append(Order(product, mid + m, LIMIT - curr_pos))
        if signal > -1 and curr_pos > 0:
            order.append(Order(product, mid - m, -curr_pos))
        return order
    
    def get_mid_price(self, depth):
        if not depth.sell_orders.keys():
            return min(depth.buy_orders.keys())
        else:
            a = max(depth.sell_orders.keys())
        if not depth.buy_orders.keys():
            return max(depth.sell_orders.keys())
        else:
            b = min(depth.buy_orders.keys())
        return (a + b) / 2


    # Define the Black-Scholes call option pricing function
    def bs_call(self, S, K, r, sigma, T):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        call_price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        return call_price

    # Define the normal cumulative distribution function
    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0



