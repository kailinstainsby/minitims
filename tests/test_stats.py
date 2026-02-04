import math
import pytest
from minitims.stats import RollingMean, RollingReturns, RollingVariance, RollingStdDev, RollingMin, RollingMax


# ============================================================================
# ROLLING MEAN TESTS
# ============================================================================

class TestRollingMeanInitialization:
    """Test RollingMean initialization."""
    
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


class TestRollingMeanUpdates:
    """Test RollingMean update behavior."""
    
    def test_update_grows_buffer_until_full(self):
        """Buffer should grow until it reaches window_size."""
        rm = RollingMean(window_size=3)
        
        rm.update(10)
        assert len(rm.buffer) == 1
        assert rm.running_sum == 10
        
        rm.update(20)
        assert len(rm.buffer) == 2
        assert rm.running_sum == 30
        
        rm.update(30)
        assert len(rm.buffer) == 3
        assert rm.running_sum == 60
    
    def test_update_evicts_oldest_when_full(self):
        """When full, new value should evict the oldest."""
        rm = RollingMean(window_size=3, initial_values=[10, 20, 30])
        rm.update(40)
        assert rm.get_window() == [20, 30, 40]
        assert rm.running_sum == 90
    
    def test_invariant_maintained_throughout(self):
        """INVARIANT: running_sum == sum(buffer) after every update."""
        rm = RollingMean(window_size=3)
        values = [5, 15, 25, 35, 45, 55]
        
        for value in values:
            rm.update(value)
            assert rm.running_sum == sum(rm.buffer), \
                f"Invariant broken after update({value}): sum={rm.running_sum}, buffer={list(rm.buffer)}"


class TestRollingMeanResults:
    """Test RollingMean mean calculation."""
    
    def test_mean_returns_none_when_not_full(self):
        """get_mean() should return None if buffer isn't full."""
        rm = RollingMean(window_size=3)
        rm.update(10)
        assert rm.get_mean() is None
        
        rm.update(20)
        assert rm.get_mean() is None
    
    def test_mean_returns_correct_value_when_full(self):
        """get_mean() should return correct average when full."""
        rm = RollingMean(window_size=3, initial_values=[10, 20, 30])
        assert rm.get_mean() == 20.0
    
    def test_mean_updates_as_window_slides(self):
        """Mean should update correctly as new values arrive."""
        rm = RollingMean(window_size=3)
        rm.update(10)
        rm.update(20)
        rm.update(30)
        assert rm.get_mean() == 20.0  # (10+20+30)/3
        
        rm.update(40)
        assert rm.get_mean() == 30.0  # (20+30+40)/3
        
        rm.update(50)
        assert rm.get_mean() == 40.0  # (30+40+50)/3


# ============================================================================
# ROLLING VARIANCE TESTS
# ============================================================================

class TestRollingVarianceInitialization:
    """Testing RollingVariance initialization."""
    
    def test_init_empty(self):
        rv = RollingVariance(window_size=2)
        assert rv.window_size == 2
        assert rv.get_window() == []
        assert len(rv.buffer) == 0
        assert rv.running_sum == 0
        assert len(rv.buffer) == 0
        assert rv.running_sum_sq == 0
        assert rv.is_full() is False

    def test_init_smaller_than_window(self):
        rv = RollingVariance(window_size=3, initial_values=[1, 2])
        assert len(rv.buffer) == 2
        assert rv.running_sum == sum(rv.buffer)
        assert rv.running_sum_sq == sum(x**2 for x in rv.buffer)

    def test_init_equal_to_window(self):
        rv = RollingVariance(window_size=3, initial_values=[1, 2, 3])
        assert len(rv.buffer) == rv.window_size
        assert rv.running_sum == sum(rv.buffer)
        assert rv.running_sum_sq == sum(x**2 for x in rv.buffer)

    def test_init_larger_than_window(self):
        rv = RollingVariance(window_size=2, initial_values=[1, 2, 3, 4])
        assert len(rv.buffer) == rv.window_size
        assert rv.get_window() == [3, 4]
        assert rv.running_sum == sum(rv.buffer)
        assert rv.running_sum_sq == sum(x**2 for x in rv.buffer)

class TestRollingVarianceUpdates:
    """Testing RollingVariance update behaviour"""

    def test_update_increments_sums(self):
        rv = RollingVariance(window_size=2)
        assert rv.running_sum == 0
        assert rv.running_sum_sq == 0

        rv.update(3)
        assert rv.running_sum == 3
        assert rv.running_sum_sq == 9

        rv.update(5)
        assert rv.running_sum == 8
        assert rv.running_sum_sq == 34

        rv.update(7)
        assert rv.running_sum == 12  # 5 + 7
        assert rv.running_sum_sq == 74  # 5^2 + 7^2

    def test_update_appends_and_pops(self):
        rv = RollingVariance(window_size=2, initial_values=[1, 2])
        assert rv.get_window() == [1, 2]
        #assert rv.running_sum == 3
        #assert rv.running_sum_sq == 5

        rv.update(3)
        assert rv.get_window() == [2, 3]
        #assert rv.running_sum == 5
        #assert rv.running_sum_sq == 13

        rv.update(4)
        assert rv.get_window() == [3, 4]
        #assert rv.running_sum == 7
        #assert rv.running_sum_sq == 25

class TestRollingVarianceResults:
    """Testing RollingVariance variance calculation."""

    def test_get_variance_returns_none_when_not_full(self):
        rv = RollingVariance(window_size=3)
        rv.update(1)
        assert rv.get_variance() is None

        rv.update(2)
        assert rv.get_variance() is None

    def test_get_variance_correct_when_full(self):
        rv = RollingVariance(window_size=3, initial_values=[2, 4, 6])
        expected_variance = 8 / 3  # Pre-calculated variance
        assert abs(rv.get_variance() - expected_variance) < 1e-10

    def test_get_variance_correct_after_updates(self):
        rv = RollingVariance(window_size=3, initial_values=[1, 2, 4])
        expected_variance = 14 / 9  # Pre-calculated variance
        assert abs(rv.get_variance() - expected_variance) < 1e-10

        rv.update(6)  # Window is now [2, 4, 6]
        expected_variance = 8 / 3
        assert abs(rv.get_variance() - expected_variance) < 1e-10

# ============================================================================
# ROLLING RETURNS TESTS
# ============================================================================

class TestRollingReturnsInitialization:
    """Test RollingReturns initialization"""

    def test_init_empty(self):
        rr= RollingReturns(window_size=3)
        assert rr.window_size == 3
        assert rr.get_window() == []
        assert len(rr.buffer) == 0
        assert rr.get_current_return() is None
        assert rr.get_mean_return() is None

    def test_init_with_single_price(self):
        """Single price should store prev_price but no returns yet"""
        rr = RollingReturns(window_size=3, initial_values=[100])
        assert rr.prev_price == 100
        assert rr.get_window() == []
        assert rr.get_current_return() is None
    
    def test_init_with_two_prices(self):
        """Two prices should compute one return"""
        rr = RollingReturns(window_size=3, initial_values=[100, 110])
        assert rr.prev_price == 110
        assert len(rr.buffer) == 1
        assert abs(rr.buffer[0] - math.log(110/100)) < 1e-10

    def test_init_with_prices(self):
        """Multiple prices should populate buffer with returns"""
        rr = RollingReturns(window_size=3, initial_values=[100, 110, 121, 133.1])
        expected_returns = [math.log(110/100), math.log(121/110), math.log(133.1/121)]
        assert rr.get_window() == expected_returns
        assert len(rr.buffer) == 3
        assert abs(rr.get_current_return() - expected_returns[-1]) < 1e-10
        assert abs(rr.get_mean_return() - (sum(expected_returns)/3)) < 1e-10

class TestRollingReturnsUpdates:
    """Test RollingReturns update behavoiour"""

    def test_update_computes_and_stores_returns(self):
        rr = RollingReturns(window_size=3)

        rr.update(100)
        assert rr.get_window() == []
        assert rr.get_current_return() is None
        assert not rr.buffer

        rr.update(110)
        assert rr.buffer[-1] == math.log(110/100) # buffer[-1] is the newest value since deque
        assert len(rr.buffer) == 1
        assert abs(rr.get_current_return() - math.log(110/100)) < 1e-10

        rr.update(121)
        assert rr.buffer[-1] == math.log(121/110)
        assert len(rr.buffer) == 2
        assert abs(rr.get_current_return() - math.log(121/110)) < 1e-10

    def test_update_evicts_oldest_when_full(self):
        rr = RollingReturns(window_size=2, initial_values=[100, 110, 121])

        assert len(rr.buffer) == 2
        assert rr.is_full() is True

        rr.update(133.1)
        assert rr.get_window() == [math.log(121/110), math.log(133.1/121)]
        assert len(rr.buffer) == 2
        assert rr.is_full() is True
        assert rr.buffer[-1] == math.log(133.1/121)
    
    def test_invariant_buffer_stores_returns_not_prices(self):
        """invariant: buffer must only contain returns, never prices"""
        rr = RollingReturns(window_size=3)
        prices = [100, 110, 121, 133.1, 146.41]
        
        for i, price in enumerate(prices):
            rr.update(price)
            # All values in buffer should be small (returns), not large (prices)
            for value in rr.buffer:
                assert abs(value) < 1.0, \
                    f"buffer contains price-like value {value} after update({price})"

class TestRollingReturnsResults:
    """Test RollingReturns results calculation"""

    def test_get_current_return(self):
        rr = RollingReturns(window_size=3)
        rr.update(100)
        assert rr.get_current_return() is None

        rr.update(110)
        expected_return = math.log(110/100)
        assert abs(rr.get_current_return() - expected_return) < 1e-10

        rr.update(121)
        expected_return = math.log(121/110)
        assert abs(rr.get_current_return() - expected_return) < 1e-10

    def test_get_mean_return_returns_none_when_not_full(self):
        """get_mean_return() should return None if buffer isn't full"""
        rr = RollingReturns(window_size=3)
        rr.update(100)
        assert rr.get_mean_return() is None
        
        rr.update(110)
        assert rr.get_mean_return() is None

    def test_get_mean_return_correct_when_full(self):
        """get_mean_return() should return correct average when full"""
        rr = RollingReturns(window_size=3)
        rr.update(100)
        rr.update(110)
        rr.update(121)
        # Buffer: [log(110/100), log(121/110)]  - NOT FULL YET (only 2 returns)
        assert rr.get_mean_return() is None
        
        rr.update(133.1)
        # Buffer: [log(110/100), log(121/110), log(133.1/121)]  - NOW FULL (3 returns)
        expected_mean = (math.log(110/100) + math.log(121/110) + math.log(133.1/121)) / 3
        assert abs(rr.get_mean_return() - expected_mean) < 1e-10
    
    def test_get_mean_return_updates_as_window_slides(self):
        """ mean return should update correctly as window slides"""
        rr = RollingReturns(window_size=2, initial_values=[100, 110, 121])
        # Buffer starts with: [log(110/100), log(121/110)]
        expected_mean = (math.log(110/100) + math.log(121/110)) / 2
        assert abs(rr.get_mean_return() - expected_mean) < 1e-10
        
        rr.update(133.1)
        # Buffer now: [log(121/110), log(133.1/121)]
        expected_mean = (math.log(121/110) + math.log(133.1/121)) / 2
        assert abs(rr.get_mean_return() - expected_mean) < 1e-10


# ============================================================================
# ROLLING STDDEV TESTS
# ============================================================================

class TestRollingStdDevResults:
    """Test RollingStdDev standard deviation calculation."""
    
    def test_stddev_returns_none_when_not_full(self):
        """get_stddev() should return None if not full."""
        rsd = RollingStdDev(window_size=3)
        rsd.update(1)
        assert rsd.get_stddev() is None
    
    def test_stddev_is_sqrt_of_variance(self):
        """StdDev should be sqrt(variance)."""
        rsd = RollingStdDev(window_size=3, initial_values=[2, 4, 6])
        variance = rsd.get_variance()
        stddev = rsd.get_stddev()
        assert abs(stddev - (variance ** 0.5)) < 1e-10
    
    def test_stddev_constant_input_is_zero(self):
        """StdDev of constant values should be 0."""
        rsd = RollingStdDev(window_size=4, initial_values=[5, 5, 5, 5])
        assert abs(rsd.get_stddev() - 0.0) < 1e-10


# ============================================================================
# ROLLING MIN/MAX TESTS
# ============================================================================

class TestRollingMinResults:
    """Test RollingMin minimum calculation."""
    
    def test_min_returns_none_when_not_full(self):
        """get_min() should return None if not full."""
        rmin = RollingMin(window_size=3)
        rmin.update(10)
        assert rmin.get_min() is None
    
    def test_min_correct_when_full(self):
        """get_min() should return correct minimum."""
        rmin = RollingMin(window_size=3, initial_values=[5, 10, 3])
        assert rmin.get_min() == 3
    
    def test_min_updates_as_window_slides(self):
        """Min should update as window slides."""
        rmin = RollingMin(window_size=3)
        rmin.update(5)
        rmin.update(10)
        rmin.update(3)
        assert rmin.get_min() == 3
        
        rmin.update(2)
        assert rmin.get_min() == 2  # [10, 3, 2]


class TestRollingMaxResults:
    """Test RollingMax maximum calculation."""
    
    def test_max_returns_none_when_not_full(self):
        """get_max() should return None if not full."""
        rmax = RollingMax(window_size=3)
        rmax.update(10)
        assert rmax.get_max() is None
    
    def test_max_correct_when_full(self):
        """get_max() should return correct maximum."""
        rmax = RollingMax(window_size=3, initial_values=[5, 10, 3])
        assert rmax.get_max() == 10
    
    def test_max_updates_as_window_slides(self):
        """Max should update as window slides."""
        rmax = RollingMax(window_size=3)
        rmax.update(5)
        rmax.update(10)
        rmax.update(3)
        assert rmax.get_max() == 10
        
        rmax.update(2)
        assert rmax.get_max() == 10  # [10, 3, 2]
        
        rmax.update(15)
        assert rmax.get_max() == 15  # [3, 2, 15]


    

    