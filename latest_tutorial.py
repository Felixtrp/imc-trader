import json
from typing import Any
from typing import Dict, List
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import copy
import jsonpickle
from collections import defaultdict, OrderedDict

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

empty_dict = {'AMETHYSTS' : 0,'STARFRUIT' : 0}
def def_value():
        return copy.deepcopy(empty_dict)

class Trader:
    
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)
    starfruit_dim = 4

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
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
                logger.print(f"Buy (condition 1): {order_for}, Product: {product}, for: {ask}")

        # Market Prices
        mprice_actual = (best_sell_pr + best_buy_pr)/2        
        mprice_ours = (acc_bid+acc_ask)/2
        
        # Undercut prices
        undercut_buy = best_buy_pr + 1 
        undercut_sell = best_sell_pr - 1 
        
        # Define the prices at which we will buy and sell
        bid_pr = min(undercut_buy, acc_bid-1) 
        sell_pr = max(undercut_sell, acc_ask+1) 
 
        # Market make buy orders
        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)            
            orders.append(Order(product, bid_pr, num))
            cpos += num
            
            logger.print(f"Buy (condition 4): {num}, Product: {product}, for: {bid_pr}")
        
        cpos = self.position[product]

        # Market take sell orders
        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
                
                logger.print(f"Sell (condition 1): {order_for}, Product: {product}, for: {bid}")

        # Market make sell orders
        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num
            logger.print(f"Sell (condition 4): {num}, Product: {product}, for: {sell_pr}")


        return orders
    
    def calc_next_price_starfruit(self, starfruit_cache):        
        coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892] 
        intercept = 4.481696494462085 
        
        nxt_price = intercept
        for i, val in enumerate(starfruit_cache):
            nxt_price += val * coef[i]
            
        return int(round(nxt_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
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
        
        
    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
      
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {'AMETHYSTS' : [], 'STARFRUIT' : []}
        conversions = 0
        trader_data = ""  
        
        
        # Update positions based on state
        for key, val in state.position.items():
            self.position[key] = val
   
        if state.traderData == "":
            starfruit_cache = []
        else: 
            starfruit_cache = jsonpickle.decode(state.traderData)
                        
        if len(starfruit_cache) == self.starfruit_dim: 
            starfruit_cache.pop(0)
            
        # best sell and best buy values obtained from order depths   
        _, bs_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)
                    
        starfruit_cache.append((bs_starfruit+bb_starfruit)/2)
        
        INF = 1e9
        
        starfruit_lb = -INF
        starfruit_ub = INF

        if len(starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit(starfruit_cache)-1
            starfruit_ub = self.calc_next_price_starfruit(starfruit_cache)+1    
            
        timestamp = state.timestamp
        
        AMETHYSTS_lb = 10000
        AMETHYSTS_ub = 10000
        
        acc_bid = {'AMETHYSTS' : AMETHYSTS_lb} 
        acc_ask = {'AMETHYSTS' : AMETHYSTS_ub} 
        acc_bid['STARFRUIT'] = starfruit_lb
        acc_ask['STARFRUIT'] = starfruit_ub
        
        for product in ['AMETHYSTS','STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
            result[product] += orders
             
        conversions = 1
        trader_data = jsonpickle.encode(starfruit_cache)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data