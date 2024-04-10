import logging
import numpy as np
import pandas as pd
import json

# Assuming necessary imports and datamodel definitions are in place

class Logger:
    def __init__(self):
        self.logs = ""

    def print(self, *args, sep=" ", end="\n"):
        self.logs += sep.join(map(str, args)) + end

    def flush(self, state, orders, conversions, trader_data):
        logging.debug(json.dumps({
            "state": state,  # Adjust this depending on the structure of 'state'
            "orders": [order.to_dict() for order in orders],
            "conversions": conversions,
            "trader_data": trader_data,
            "logs": self.logs
        }, indent=4))
        self.logs = ""

class Trader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.alpha = 0.3  # Optimal alpha for EWMA found from analysis
        self.pos_lim = {"STARFRUIT": 20, "AMETHYSTS": 20}
        logging.info("Trader initialized.")

    def load_data(self):
        try:
            logging.info(f"Loading data from {self.filepath}")
            return pd.read_csv(self.filepath, sep=';')
        except FileNotFoundError as e:
            logging.error(f"File not found: {self.filepath}")
            raise e
        except Exception as e:
            logging.error("Error loading data")
            raise e

    def run(self):
        logging.info("Starting run method.")
        df = self.load_data()
        # Continue implementation of trading logic here

# Setup logging
logging.basicConfig(filename='trading_log.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    trader = Trader(r'D:\IMC 比赛\imc-trader\Round1\round-1-island-data-bottle\prices_round_1_day_0.csv')
    trader.run()
