import json
from typing import Any
from typing import Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import copy
import jsonpickle
from collections import OrderedDict
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

empty_dict = {'AMETHYSTS' : 0,
              'STARFRUIT' : 0,
              'ORCHIDS' : 0,
              'STRAWBERRIES' : 0,
              'CHOCOLATE': 0,
              'ROSES' : 0,
              'GIFT_BASKET' : 0,
              'COCONUT': 0,
              'COCONUT_COUPON': 0,
              }

def def_value():
    return copy.deepcopy(empty_dict)

class Trader:
    timestamp = 0
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20,
                      'STARFRUIT' : 20,
                      'ORCHIDS' : 100,
                      'STRAWBERRIES' : 350,
                      'CHOCOLATE': 250,
                      'ROSES' : 60,
                      'GIFT_BASKET' : 60,
                      'COCONUT': 300,
                      'COCONUT_COUPON': 600
                      }

    max_starfruit_length = 20

    basket_std = 77
    basket_premium = 380

    trading_days = 250
    day_number = 4
    timestamps_per_day = 1000000
    coconut_annualised_volatility = 0.00010095 * (trading_days * (timestamps_per_day / 100))**(1/2)
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
    
    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        # Ordered depths of buy and sell orders
        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # Find the highest ask and lowest bid and the total volumes
        sell_vol, best_sell_pr = self.values_extract(osell) 
        buy_vol, best_buy_pr = self.values_extract(obuy, 1) 
        
        # Current position in the product
        cpos = self.position[product]

        mx_with_buy = -1

        # Market take buy orders
        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask) 
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos) 
                cpos += order_for # 
                assert(order_for >= 0)
                
                orders.append(Order(product, ask, order_for)) 

        # Market Prices
        mprice_actual = (best_sell_pr + best_buy_pr)/2        
        mprice_ours = (acc_bid+acc_ask)/2
        
        # Undercut prices
        buy_tail, sell_tail = 1, 1
        if cpos > 15:
            buy_tail = 0
            sell_tail = 2
        elif cpos < -15:
            buy_tail = 2
            sell_tail = 0
        undercut_buy = best_buy_pr + buy_tail
        undercut_sell = best_sell_pr - sell_tail
        
        # Define the prices at which we will buy and sell
        bid_pr = min(undercut_buy, acc_bid-1) 
        sell_pr = max(undercut_sell, acc_ask+1) 
 
        # Market make buy orders
        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)            
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        # Market take sell orders
        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        # Market make sell orders
        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def calc_next_price_starfruit(self, mid_prices):
        log_returns = np.log(mid_prices/np.roll(mid_prices, shift=1))[1:]
        len = log_returns.size
        alpha = 0.3
        weights = np.power((1 - alpha)*np.ones(shape=len), np.arange(len))
        predicted_log_return = np.sum(weights*log_returns) / np.sum(weights)
        predicted_mid_price = np.exp(predicted_log_return + np.log(mid_prices[-1]))

        return int(round(predicted_mid_price))
    
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        
        limit = self.POSITION_LIMIT[product]

        _, best_sell_pr = self.values_extract(osell)
        _, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]
        
        # market take
        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < limit:
                order_for = min(-vol, limit - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, int(ask), order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) 
        sell_pr = max(undercut_sell, acc_ask)

        # market make
        if cpos < limit:
            num = limit - cpos
            orders.append(Order(product, int(bid_pr), num))
            cpos += num
        
        cpos = self.position[product]
        
        # market take
        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -limit:
                order_for = max(-vol, -limit-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, int(bid), order_for))

        # market make
        if cpos > -limit:
            num = -limit-cpos
            orders.append(Order(product, int(sell_pr), num))
            cpos += num

        return orders

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
                return [], []
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

    def volatility_smile(self, stock_price):
        a, c = 9.510469372524675e-08, 0.15885082971839998
        return a*(stock_price - self.coconut_option_strike)**2 + c
    
    def normal_cdf(self, x):
        q = math.erf(x / math.sqrt(2.0))
        return (1.0 + q) / 2.0

    def black_scholes(self, stock_mid_price):
        time_to_expiry_beginning = 1 - (self.day_number-1)/self.trading_days
        time_to_expiry = time_to_expiry_beginning - (self.timestamp / (self.trading_days*self.timestamps_per_day))
        ann_vol = self.volatility_smile(stock_mid_price) 
        vol_adjustment = ann_vol * math.sqrt(time_to_expiry)
        d_plus = (1/vol_adjustment) * (math.log(stock_mid_price / self.coconut_option_strike) + ((ann_vol**2)/2)*time_to_expiry)
        d_minus = d_plus - vol_adjustment
        fair_value = self.normal_cdf(d_plus)*stock_mid_price - self.normal_cdf(d_minus)*self.coconut_option_strike

        return fair_value
    
    def compute_orders_coconut_coupon(self, state):
        order_depth = state.order_depths
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

        residual_sell = best_bids['COCONUT_COUPON'] - self.black_scholes(best_asks['COCONUT'])
        residual_buy = best_asks['COCONUT_COUPON'] - self.black_scholes(best_bids['COCONUT'])

        trade_at = self.residual_std*0.3

        if residual_sell > trade_at:
            available_volume = self.position['COCONUT_COUPON'] + self.POSITION_LIMIT['COCONUT_COUPON']
            buy_orders = order_depth["COCONUT_COUPON"].buy_orders.items()
            for price, quantity in buy_orders:
                if available_volume <= 0:
                    break
                else:
                    volume = -min(quantity, available_volume)
                    # Sell order
                    orders.append(Order('COCONUT_COUPON', price, volume))
                    available_volume += volume

        if residual_buy < -trade_at:
            available_volume = self.POSITION_LIMIT['COCONUT_COUPON'] - self.position['COCONUT_COUPON']
            sell_orders = order_depth["COCONUT_COUPON"].sell_orders.items()
            for price, quantity in sell_orders:
                if available_volume <= 0:
                    break
                else:
                    volume = min(available_volume, -quantity)
                    # Buy order
                    orders.append(Order('COCONUT_COUPON', price, volume))
                    available_volume -= volume

        return orders
      
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {'AMETHYSTS' : [],
                  'STARFRUIT' : [],
                  'ORCHIDS' : [],
                  'GIFT_BASKET' : [],
                  'COCONUT': [],
                  'COCONUT_COUPON': [],
                  }
        conversions = 0
        trader_data = ""  
        
        self.timestamp = state.timestamp

        # Update positions based on state
        for key, val in state.position.items():
            self.position[key] = val

        if state.traderData == "":
            starfruit_cache = []
        else:
            starfruit_cache = jsonpickle.decode(state.traderData)

        # best sell and best buy values obtained from order depths
        if bool(state.order_depths["STARFRUIT"].sell_orders) and bool(state.order_depths["STARFRUIT"].buy_orders):
            _, bs_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
            _, bb_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)        
            starfruit_mid_price = (bs_starfruit+bb_starfruit)/2
            starfruit_cache.append(starfruit_mid_price)

        if len(starfruit_cache) > self.max_starfruit_length:
            starfruit_cache.pop(0)

        mid_prices = np.array(starfruit_cache)

        INF = 1e9
        
        starfruit_lb = -INF
        starfruit_ub = INF

        if mid_prices.size > 1:
            starfruit_lb = self.calc_next_price_starfruit(mid_prices)-1
            starfruit_ub = self.calc_next_price_starfruit(mid_prices)+1    
                
        AMETHYSTS_lb = 10000
        AMETHYSTS_ub = 10000
        
        acc_bid = {'AMETHYSTS' : AMETHYSTS_lb} 
        acc_ask = {'AMETHYSTS' : AMETHYSTS_ub} 
        acc_bid['STARFRUIT'] = starfruit_lb
        acc_ask['STARFRUIT'] = starfruit_ub
        obs = state.observations.conversionObservations['ORCHIDS']
        acc_bid['ORCHIDS'] = obs.bidPrice
        acc_ask['ORCHIDS'] = obs.askPrice

        result['AMETHYSTS'] += self.compute_orders_amethysts("AMETHYSTS", state.order_depths["AMETHYSTS"], acc_bid["AMETHYSTS"], acc_ask["AMETHYSTS"])
        result['STARFRUIT'] += self.compute_orders_regression("STARFRUIT", state.order_depths["STARFRUIT"], acc_bid["STARFRUIT"], acc_ask["STARFRUIT"])

        conversions, orders = self.compute_orders_orchids("ORCHIDS", state.order_depths["ORCHIDS"], state, acc_bid['ORCHIDS'], acc_ask['ORCHIDS'])
        result['ORCHIDS'] += orders

        result['GIFT_BASKET'] += self.compute_orders_gift_baskets(state)
        
        result['COCONUT_COUPON'] += self.compute_orders_coconut_coupon(state)

        trader_data = jsonpickle.encode(starfruit_cache)
        logger.flush(state, result, conversions, "")
        return result, conversions, trader_data