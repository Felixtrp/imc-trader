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

empty_dict = {'COCONUT' : 0,
              'COCONUT_COUPON': 0
              }
def def_value():
        return copy.deepcopy(empty_dict)

class Trader:

    timestamp = 0
    def __init__(self) -> None:
        self.last_pos_orchids = 0
    
    position = copy.deepcopy(empty_dict)
    volume_traded = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20,
                      'STARFRUIT' : 20,
                      'ORCHIDS' : 100,
                      'STRAWBERRIES' : 350,
                      'CHOCOLATE': 250,
                      'ROSES' : 60,
                      'GIFT_BASKET' : 60,
                      'COCONUT_COUPON': 600
                      }
    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)
    starfruit_dim = 4

    basket_std = 77
    basket_premium = 380

    trading_days = 250
    day_number = 1
    timestamps_per_day = 1000000
    coconut_annualised_volatility = 0.000101 * (trading_days * (timestamps_per_day / 100))**(1/2)
    coconut_option_strike = 10000
    residual_std = 13.3
    
    def best_ask(self, order_depth):
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        if bool(sell_orders):
            return next(iter(sell_orders))
        else:
            return None
    
    def best_bid(self, order_depth):
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        if bool(buy_orders):
            return next(iter(buy_orders))
        else:
            return None
    
    def normal_cdf(self, x):
        q = math.erf(x / math.sqrt(2.0))
        return (1.0 + q) / 2.0

    def black_scholes(self, stock_mid_price):
        time_to_expiry_beginning = 1 - (self.day_number-1)/self.trading_days
        time_to_expiry = time_to_expiry_beginning - (self.timestamp / (self.trading_days*self.timestamps_per_day))
        vol_adjustment = self.coconut_annualised_volatility * math.sqrt(time_to_expiry)
        d_plus = (1/vol_adjustment) * (math.log(stock_mid_price / self.coconut_option_strike) + ((self.coconut_annualised_volatility**2)/2)*time_to_expiry)
        d_minus = d_plus - vol_adjustment
        fair_value = self.normal_cdf(d_plus)*stock_mid_price - self.normal_cdf(d_minus)*self.coconut_option_strike

        return fair_value

    def compute_orders_coconut_coupon(self, order_depth):
        orders : List[Order] = []
        prods = ['COCONUT', 'COCONUT_COUPON']
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

        coupon_fair_value = self.black_scholes(mid_prices['COCONUT'])
        residual_sell = best_bids['COCONUT_COUPON'] - self.black_scholes(best_asks['COCONUT'])
        residual_buy = best_asks['COCONUT_COUPON'] - self.black_scholes(best_bids['COCONUT'])

        trade_at = self.residual_std*0.3

        if residual_sell > trade_at:
            available_volume = self.position['COCONUT_COUPON'] + self.POSITION_LIMIT['COCONUT_COUPON']
            buy_orders = order_depth["COCONUT_COUPON"].buy_orders.items()
            assert(available_volume >= 0)
            for price, quantity in buy_orders:
                if available_volume <= 0:
                    break
                else:
                    orders.append(Order('COCONUT_COUPON', price, -quantity))
                    available_volume -= quantity

        elif residual_buy < -trade_at:
            available_volume = self.POSITION_LIMIT['COCONUT_COUPON'] - self.position['COCONUT_COUPON']
            sell_orders = order_depth["COCONUT_COUPON"].sell_orders.items()
            assert(available_volume >= 0)
            for price, quantity in sell_orders:
                if available_volume <= 0:
                    break
                else:
                    orders.append(Order('COCONUT_COUPON', price, -quantity))
                    available_volume += quantity

        return orders
      
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {'COCONUT_COUPON': []}
        conversions = 0
        trader_data = ""  
        
        # Update positions based on state
        for key, val in state.position.items():
            self.position[key] = val

        self.timestamp = state.timestamp

        result['COCONUT_COUPON'] += self.compute_orders_coconut_coupon(state.order_depths)

        logger.flush(state, result, conversions, "")
        return result, conversions, trader_data