import pytest
from minitims.stats import RollingMean

class TestRollingMeanInitialization:
    def test_init_empty(self):
        rm = RollingMean(window_size=3)
        assert rm.window_size == 3
        assert rm.get_window() == []
        assert len(rm.buffer) == 0
        assert rm.running_sum == 0 

    def test_init_with_values_less_than_window(self):
        rm = RollingMean(window_size=4, initial_values=[1, 2])
        assert rm.window_size == 4
        assert rm.get_window() == [1, 2]
        assert len(rm.buffer) == 2
        assert rm.running_sum == 3

    def test_init_with_values_equal_to_window(self):
        rm = RollingMean(window_size=3, initial_values=[1, 2, 3])
        assert rm.window_size == 3
        assert rm.get_window() == [1, 2, 3]
        assert len(rm.buffer) == 3
        assert rm.running_sum == 6

    def test_init_with_values_greater_than_window(self):
        rm = RollingMean(window_size=2, initial_values=[1, 2, 3, 4])
        assert rm.window_size == 2
        assert rm.get_window() == [3, 4]
        assert len(rm.buffer) == 2
        assert rm.running_sum == 7