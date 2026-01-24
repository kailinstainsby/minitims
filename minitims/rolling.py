from collections import deque

class RollingWindow:
    def __init__(self, window_size: int, initial_values=None) -> None:
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

        if initial_values:
            start_index = max(0, len(initial_values) - window_size)
            for value in initial_values[start_index:]:
                self.buffer.append(value)

    def update(self, new_value) -> None:
        self.buffer.append(new_value)

    def is_full(self) -> bool:
        return len(self.buffer) == self.window_size

    def get_window(self) -> list:
        return list(self.buffer)

