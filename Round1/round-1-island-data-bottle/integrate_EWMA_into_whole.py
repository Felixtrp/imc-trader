import json
import time
import numpy as np
import pandas as pd
from datamodel import Listing, Observation, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, Order 

class Logger:
    def __init__(self):
        self.logs = ""

    def print(self, *args, sep=" ", end="\n"):
        self.logs += sep.join(map(str, args)) + end

    def flush(self, state, orders, conversions, trader_data):
        print(json.dumps({
            "state": state.to_dict(),  # Convert state to a dictionary for logging
            "orders": [order.to_dict() for order in orders],
            "conversions": conversions,
            "trader_data": trader_data,
            "logs": self.logs
        }, cls=ProsperityEncoder))  # Ensure you have a JSON encoder for custom objects
        self.logs = ""

class Trader:
    def __init__(self):
        self.pos_lim = {"STARFRUIT": 20, "AMETHYSTS": 20}
        self.alpha = 0.3  # Optimal alpha for EWMA found from analysis

    def calculate_ewma(self, prices, alpha):
        """ Calculate Exponentially Weighted Moving Average """
        if not prices:
            return None
        return pd.Series(prices).ewm(alpha=alpha, adjust=False).mean().iloc[-1]

    def trade(self, product, current_price, predicted_price, current_position, order_depth):
        orders = []
        if predicted_price is None:
            return orders  # No prediction available, skip trading logic

        for price, volume in order_depth.items():
            if price < predicted_price and current_position < self.pos_lim[product]:
                # Buy condition for under predicted price
                buy_volume = min(volume, self.pos_lim[product] - current_position)
                orders.append(Order(symbol=product, price=price, quantity=buy_volume))
                current_position += buy_volume
            elif price > predicted_price and current_position > -self.pos_lim[product]:
                # Sell condition for over predicted price
                sell_volume = min(volume, current_position + self.pos_lim[product])
                orders.append(Order(symbol=product, price=price, quantity=-sell_volume))
                current_position -= sell_volume
        return orders, current_position

    def process_data(self, filepath):
        df = pd.read_csv(filepath)
        df['weighted_price'] = ((df['bid_price_1'] * df['bid_volume_1']) + (df['ask_price_1'] * df['ask_volume_1'])) / (df['bid_volume_1'] + df['ask_volume_1'])
        df['returns'] = df['weighted_price'].pct_change()
        return df

    def run(self, state):
        logger = Logger()
        df = self.process_data(state.filepath)
        
        starfruit_data = df[df['product'] == 'STARFRUIT']
        starfruit_prices = starfruit_data['weighted_price'].tolist()
        starfruit_predicted_price = self.calculate_ewma(starfruit_prices, self.alpha)

        amethysts_data = df[df['product'] == 'AMETHYSTS']
        amethysts_prices = amethysts_data['weighted_price'].tolist()
        amethysts_predicted_price = amethysts_prices[-1] if amethysts_prices else None  # Assuming stable

        current_positions = {
            "STARFRUIT": state.position.get('STARFRUIT', 0),
            "AMETHYSTS": state.position.get('AMETHYSTS', 0)
        }

        starfruit_orders, new_starfruit_pos = self.trade(
            "STARFRUIT", starfruit_prices[-1], starfruit_predicted_price, current_positions["STARFRUIT"], state.order_depth["STARFRUIT"])
        amethysts_orders, new_amethysts_pos = self.trade(
            "AMETHYSTS", amethysts_prices[-1], amethysts_predicted_price, current_positions["AMETHYSTS"], state.order_depth["AMETHYSTS"])

        orders = {
            "STARFRUIT": starfruit_orders,
            "AMETHYSTS": amethysts_orders
        }

        # Update trader data for persistence and analysis
        trader_data = json.dumps({
            "STARFRUIT_prices": starfruit_prices,  # Save new positions for continued strategy application
            "AMETHYSTS_prices": amethysts_prices
        })

        logger.print("Trading cycle complete")
        logger.flush(state, orders, 0,
