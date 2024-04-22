import json
from typing import Any
from typing import Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import copy
import jsonpickle
from collections import defaultdict, OrderedDict
import collections
import math
import numpy as np

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

empty_dict = {'GIFT_BASKET' : 0}
def def_value():
        return copy.deepcopy(empty_dict)

class Trader:

    def __init__(self) -> None:
        self.last_pos_orchids = 0
    
    position = copy.deepcopy(empty_dict)
    volume_traded = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100, 'STRAWBERRIES' : 350, 'CHOCOLATE': 250, 'ROSES' : 60, 'GIFT_BASKET' : 60}
    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)
    starfruit_dim = 4
    basket_std = 77
    basket_premium = 380
    
    def best_ask(self, order_depth):
        return next(iter(OrderedDict(sorted(order_depth.sell_orders.items()))))
    
    def best_bid(self, order_depth):
        return next(iter(OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))))
    
    def compute_orders_gift_baskets(self, state):
        order_depth = state.order_depths
        orders = []
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        best_asks = {}
        best_bids = {}
        mid_prices = {}

        for p in prods:
            best_ask = self.best_ask(order_depth[p])
            best_bid = self.best_bid(order_depth[p])
            if best_ask == None or best_bid == None:
                return []
            else:
                best_asks[p] = best_ask
                best_bids[p] = best_bid
                mid_prices[p] = (1/2)*(best_ask + best_bid)

        residual_sell = best_bids['GIFT_BASKET']  - best_asks['STRAWBERRIES']*6 - best_asks['CHOCOLATE']*4 - best_asks['ROSES'] - self.basket_premium
        residual_buy = best_asks['GIFT_BASKET']  - best_bids['STRAWBERRIES']*6 - best_bids['CHOCOLATE']*4 - best_bids['ROSES'] - self.basket_premium

        trade_at = self.basket_std*0.4

        if residual_sell > trade_at:
            available_volume = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            buy_orders = order_depth["GIFT_BASKET"].buy_orders.items()
            for price, quantity in buy_orders:
                if available_volume <= 0:
                    break
                else:
                    volume = -min(quantity, available_volume)
                    # Sell order
                    orders.append(Order('GIFT_BASKET', price, volume))
                    available_volume += volume

        if residual_buy < -trade_at:
            available_volume = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            sell_orders = order_depth["GIFT_BASKET"].sell_orders.items()
            for price, quantity in sell_orders:
                if available_volume <= 0:
                    break
                else:
                    volume = min(available_volume, -quantity)
                    # Buy order
                    orders.append(Order('GIFT_BASKET', price, volume))
                    available_volume -= volume

        return orders
      
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {'GIFT_BASKET': [], 'CHOCOLATE': [], 'ROSES' : [], 'STRAWBERRIES' :[]}
        conversions = 0
        trader_data = ""  
        
        # Update positions based on state
        for key, val in state.position.items():
            self.position[key] = val

        result['GIFT_BASKET'] += self.compute_orders_gift_baskets(state)

        logger.flush(state, result, conversions, "")
        return result, conversions, trader_data