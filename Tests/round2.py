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

    def compute_orders_orchids(self, product, order_depth, state, acc_bid, acc_ask):
        orders: list[Order] = []

        limit = self.POSITION_LIMIT[product]
        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        cpos = self.position[product]

        obs = state.observations.conversionObservations[product]
        import_tariff = obs.importTariff
        shipment_cost = obs.transportFees
        mid_price = (acc_ask + acc_bid) / 2
        ask = int(acc_ask + shipment_cost + import_tariff)
        did_sell = 0
        conversions = -cpos
        if math.ceil(mid_price) < min(obuy, key=obuy.get):
            did_sell, orders = self.market_taker_sell(product, orders, obuy, 0, ask + import_tariff + shipment_cost, limit, did_sell) #cpos
        if int(mid_price) - 1 > acc_ask + import_tariff + shipment_cost:
            orders.append(Order(product, int(mid_price) - 1, -limit - did_sell))
        elif int(mid_price) - 0 > acc_ask + import_tariff + shipment_cost:
            orders.append(Order(product, int(mid_price) - 0, -limit - did_sell))
        return conversions, orders
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {'ORCHIDS': []}
        conversions = 0
        trader_data = ""  

        # Update positions based on state
        for key, val in state.position.items():
            self.position[key] = val

        obs = state.observations.conversionObservations['ORCHIDS']

        acc_bid = {}
        acc_ask = {}
        acc_bid['ORCHIDS'] = obs.bidPrice
        acc_ask['ORCHIDS'] = obs.askPrice

        conversions, orders = self.compute_orders_orchids("ORCHIDS", state.order_depths["ORCHIDS"], state, acc_bid['ORCHIDS'], acc_ask['ORCHIDS'])
        result['ORCHIDS'] += orders

        logger.flush(state, result, conversions, "")
        return result, conversions, trader_data