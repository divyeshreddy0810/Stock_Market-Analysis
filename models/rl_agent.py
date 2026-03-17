import numpy as np

class RLAgent:
    """Simple Q-Learning style RL agent for Buy/Hold/Sell recommendation"""

    def __init__(self, buy_threshold=0.02, sell_threshold=-0.02):
        """
        Thresholds are % expected return to trigger buy/sell
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def recommend(self, current_price, forecasted_price):
        """
        current_price: last known price
        forecasted_price: forecast N days ahead (last forecast value)
        Returns: "Buy", "Hold", "Sell"
        """
        expected_return = (forecasted_price - current_price) / current_price
        if expected_return >= self.buy_threshold:
            return "Buy"
        elif expected_return <= self.sell_threshold:
            return "Sell"
        else:
            return "Hold"
