from trading_system import TradingSystem
from alpaca_connection import AlpacaPaperSocket


class PortfolioManagementSystem(TradingSystem):

    def __init__(self):
        super().__init__(AlpacaPaperSocket(), 'IBM', 604800, 1, 'AI_PM')

    def place_buy_order(self):
        pass

    def place_sell_order(self):
        pass

    def system_loop(self):
        pass
