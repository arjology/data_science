import time
import numpy as np
import pandas as pd

from alpaca_connection import AlpacaPaperSocket
from AI_PM import PortfolioManagementModel
from trading_system import TradingSystem
from algo import base_url, api_key_id, api_secret


class PortfolioManagementSystem(TradingSystem):
    def __init__(self):
        super().__init__(
            AlpacaPaperSocket(
                key_id=api_key_id, secret_key=api_secret, base_url=base_url
            ),
            "IBM",
            86400,
            1,
            "AI_PM",
        )
        self.AI = PortfolioManagementModel()

    def place_buy_order(self):
        self.api.submit_order(
            symbol="IBM", qty=1, side="buy", type="market", time_in_force="day"
        )

    def place_sell_order(self):
        self.api.submit_order(
            symbol="IBM", qty=1, side="sell", type="market", time_in_force="day"
        )

    def system_loop(self):
        # Variables for weekly close
        this_weeks_close = 0
        last_weeks_close = 0
        delta = 0
        day_count = 0
        while True:
            # Wait a day to request more data
            time.sleep(1440)
            # Request EoD data for IBM
            data_req = self.api.get_barset("IBM", timeframe="1D", limit=1).df
            # Construct dataframe to predict
            x = pd.DataFrame(
                data=[[data_req["IBM"]["close"][0]]], columns="Close".split()
            )
            if day_count == 7:
                day_count = 0
                last_weeks_close = this_weeks_close
                this_weeks_close = x["Close"]
                delta = this_weeks_close - last_weeks_close

                # AI choosing to buy, sell, or hold
                if np.around(self.AI.network.predict([delta])) <= -0.5:
                    self.place_sell_order()

                elif np.around(self.AI.network.predict([delta]) >= 0.5):
                    self.place_buy_order()


if __name__ == "__main__":
    PortfolioManagementSystem()
