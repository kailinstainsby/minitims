import math
from .rolling import RollingWindow

class RollingMean(RollingWindow): # O(1) rolling mean calculation
    def __init__(self, window_size: int, initial_values=None) -> None:
        # Call parent class to set up buffer
        super().__init__(window_size, initial_values)

        # Initialize running_sum
        # If we have initial values, sum them up, else running_sum is 0
        self.running_sum = sum(self.buffer)

    def update(self, new_value) -> None:
        # if the buffer is full, we remove the oldest value
        if self.is_full():
            # Subtract the value that will be removed
            self.running_sum -= self.buffer[0]

        # Add the new value to the running sum
        self.running_sum += new_value

        #Update the superclass buffer
        super().update(new_value)

    def get_mean(self) -> float | None:
        if self.is_full():
            return self.running_sum / self.window_size
        return None

class RollingVariance(RollingWindow): # O(1) rolling variance calculation
    def __init__(self, window_size: int, initial_values=None) -> None:
        # Call parent class to set up buffer
        super().__init__(window_size, initial_values)

        self.running_sum = sum(self.buffer)
        self.running_sum_sq = sum(x**2 for x in self.buffer)

    def update(self, new_value) -> None:
        if self.is_full():
            self.running_sum -= self.buffer[0]
            self.running_sum_sq -= self.buffer[0] ** 2
        
        self.running_sum += new_value
        self.running_sum_sq += new_value ** 2
        super().update(new_value)

    def get_variance(self) -> float | None:
        if self.is_full():
            mean = self.running_sum / self.window_size
            mean_sq = self.running_sum_sq / self.window_size
            return mean_sq - (mean ** 2)
        return None
    
class RollingReturns(RollingWindow): # O(1) rolling returns calculation
    # computes log returns from a stream of prices
    def __init__(self, window_size: int, initial_values=None) -> None:
        super().__init__(window_size, initial_values=None)
        
        # Track the previous price for computing returns
        self.prev_price = None
        
        # If we have initial prices, convert them to returns and populate buffer
        if initial_values and len(initial_values) > 1:
            # First price is just stored as prev_price
            self.prev_price = initial_values[0]
            
            # Convert remaining prices to returns and store in buffer
            for price in initial_values[1:]:
                ret = math.log(price / self.prev_price)
                self.buffer.append(ret)
                self.prev_price = price
        elif initial_values and len(initial_values) == 1:
            # Only one price - just store it, no returns yet
            self.prev_price = initial_values[0]

    def update(self, new_price: float) -> None:
        # feed in a new price, compute return, store return in buffer
        if self.prev_price is not None:

            # compute log return; log(P_t / P_{t-1})
            ret = math.log(new_price / self.prev_price)
            
            # store the return (not the price) in the buffer
            super().update(ret)
        
        # update prev_price for next iteration
        self.prev_price = new_price
    
    def get_current_return(self) -> float | None:
        # get the most recent return
        if len(self.buffer) > 0:
            return self.buffer[-1]
        return None
    
    def get_mean_return(self) -> float | None:
        # get average of returns in the window (only when full)
        if self.is_full():
            return sum(self.buffer) / self.window_size
        return None

    
class RollingStdDev(RollingVariance): # O(1) rolling stddev calculation
    def get_stddev(self) -> float | None:
        variance = self.get_variance()
        if variance is not None:
            return variance ** 0.5
        return None

class RollingMin(RollingWindow): # O(n) rolling min calculation
    def __init__(self, window_size: int, initial_values=None) -> None:
        super().__init__(window_size, initial_values)

    def update(self, new_value) -> None:
        super().update(new_value)

    def get_min(self) -> float | None:
        if self.is_full():
            return min(self.buffer)
        return None
    
class RollingMax(RollingWindow): # O(n) rolling max calculation
    def __init__(self, window_size: int, initial_values=None) -> None:
        super().__init__(window_size, initial_values)

    def update(self, new_value) -> None:
        super().update(new_value)

    def get_max(self) -> float | None:
        if self.is_full():
            return max(self.buffer)
        return None