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

empty_dict = {'AMETHYSTS' : 0,'STARFRUIT' : 0, 'ORCHIDS' : 0, 'STRAWBERRIES' : 0, 'CHOCOLATE': 0, 'ROSES' : 0, 'GIFT_BASKET' : 0}
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
    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0
    basket_std = 77
    
    def best_ask(self, order_depth):
        return next(iter(OrderedDict(sorted(order_depth.sell_orders.items()))))
    
    def best_bid(self, order_depth):
        return next(iter(OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))))

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
    
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]
        
        # market take
        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, int(ask), order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) 
        sell_pr = max(undercut_sell, acc_ask)

        # market make
        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, int(bid_pr), num))
            cpos += num
        
        cpos = self.position[product]
        
        # market take
        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, int(bid), order_for))

        # market make
        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, int(sell_pr), num))
            cpos += num

        return orders
    
    def compute_orders_orchids(self, product, order_depth, state, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

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
            did_sell, orders = self.market_taker_sell(product, orders, obuy, 0, ask + import_tariff + shipment_cost, LIMIT, did_sell) #cpos
        if int(mid_price) - 1 > acc_ask + import_tariff + shipment_cost:
            orders.append(Order(product, int(mid_price) - 1, -LIMIT - did_sell))
        elif int(mid_price) - 0 > acc_ask + import_tariff + shipment_cost:
            orders.append(Order(product, int(mid_price) - 0, -LIMIT - did_sell))
        return conversions, orders
    
    #def compute_orders_gift_baskets(product, order_depth, state, underlying_bid, underlying_ask, limit):
    def compute_orders_gift_baskets(self, order_depth):

        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 380
        res_sell = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 380

        trade_at = self.basket_std*0.5
        close_at = self.basket_std*(-1000)

        pb_pos = self.position['GIFT_BASKET']
        pb_neg = self.position['GIFT_BASKET']

        s_pos = self.position['STRAWBERRIES']
        r_pos = self.position['ROSES']
        c_pos = self.position['CHOCOLATE']

        if self.position['GIFT_BASKET'] == 60:
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -350 - s_pos))
            orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -250 - c_pos))
            orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -60 - r_pos))
        if self.position['GIFT_BASKET'] == -60:
            orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], 350 - s_pos))
            orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], 250 - c_pos))
            orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], 60 - r_pos))

        basket_buy_sig = 0
        basket_sell_sig = 0

        if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_buy_basket_unfill = 0
        if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], min(350, 6*vol)))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], min(250, 4*vol)))
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], max(-350, -6*vol)))
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], max(-250, -4*vol)))
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol))
                self.cont_buy_basket_unfill += 2
                pb_pos += vol


        return orders
    def compute_orders(self, product, order_depth, state, acc_bid, acc_ask):
        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        if product == 'ORCHIDS':
            return self.compute_orders_orchids(product, order_depth, state, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        if product == 'GIFT_BASKET':
            #return self.compute_orders_gift_baskets(product, order_depth, state, acc_bid, acc_ask, self.POSITION_LIMIT[product])
            return self.compute_orders_gift_baskets(state.order_depths)
      
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS': [], 'GIFT_BASKET': [], 'CHOCOLATE': [], 'ROSES' : [], 'STRAWBERRIES' :[]}
        conversions = 0
        trader_data = ""  
        
        
        # Update positions based on state
        for key, val in state.position.items():
            self.position[key] = val

        #if state.traderData == "":
        #    starfruit_cache = []
        #else:
        #    starfruit_cache = jsonpickle.decode(state.traderData)

        # best sell and best buy values obtained from order depths   
        #_, bs_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        #_, bb_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)        

        #starfruit_mid_price = (bs_starfruit+bb_starfruit)/2

        #starfruit_cache.append(starfruit_mid_price)

        #if len(starfruit_cache) > 20:
        #    starfruit_cache.pop(0)

        #mid_prices = np.array(starfruit_cache)

        INF = 1e9
        
        starfruit_lb = -INF
        starfruit_ub = INF

        #if mid_prices.size > 1:
        #    starfruit_lb = self.calc_next_price_starfruit(mid_prices)-1
        #    starfruit_ub = self.calc_next_price_starfruit(mid_prices)+1    
            
        timestamp = state.timestamp
        
        AMETHYSTS_lb = 10000
        AMETHYSTS_ub = 10000
        
        acc_bid = {'AMETHYSTS' : AMETHYSTS_lb} 
        acc_ask = {'AMETHYSTS' : AMETHYSTS_ub} 
        #acc_bid['STARFRUIT'] = starfruit_lb
        #acc_ask['STARFRUIT'] = starfruit_ub

        #obs = state.observations.conversionObservations['ORCHIDS']

        #acc_bid['ORCHIDS'] = obs.bidPrice
        #acc_ask['ORCHIDS'] = obs.askPrice

        chocolate_ask = self.best_ask(state.order_depths['CHOCOLATE'])
        chocolate_bid = self.best_bid(state.order_depths['CHOCOLATE'])
        strawberries_ask = self.best_ask(state.order_depths['STRAWBERRIES'])
        strawberries_bid = self.best_bid(state.order_depths['STRAWBERRIES'])
        roses_ask = self.best_ask(state.order_depths['ROSES'])
        roses_bid = self.best_bid(state.order_depths['ROSES'])

        acc_bid['GIFT_BASKET'] = 4*chocolate_ask + 6*strawberries_ask + roses_ask
        acc_ask['GIFT_BASKET'] = 4*chocolate_bid + 6*strawberries_bid + roses_bid

        orders = self.compute_orders_gift_baskets(state.order_depths)
        result['GIFT_BASKET'] += orders['GIFT_BASKET']
        result['ROSES'] += orders['ROSES']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        
        #for product in ['AMETHYSTS','STARFRUIT', 'ORCHIDS', 'GIFT_BASKET']:
        for product in ['GIFT_BASKET']:
            order_depth: OrderDepth = state.order_depths[product]
            if product == 'AMETHYSTS' or product == 'STARFRUIT':
                result[product] = self.compute_orders(product, order_depth, state, acc_bid[product], acc_ask[product])
            #elif product == 'GIFT_BASKET':
            #    result[product] = self.compute_orders(product, order_depth, state, acc_bid[product], acc_ask[product])
            elif product == 'ORCHIDS':
                conversions, result[product] = self.compute_orders(product, order_depth, state, acc_bid[product], acc_ask[product])

        #trader_data = jsonpickle.encode(starfruit_cache)
        logger.flush(state, result, conversions, "")
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
