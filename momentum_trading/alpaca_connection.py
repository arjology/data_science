from alpaca_trade_api import REST


class AlpacaPaperSocket(REST):
    def __init__(self, key_id: str, secret_key: str, base_url: str):
        super().__init__(key_id=key_id, secret_key=secret_key, base_url=base_url)
