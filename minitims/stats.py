from rolling import RollingWindow

class RollingMean(RollingWindow):
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