import pytest
from minitims.stats import RollingMean, RollingVariance, RollingStdDev, RollingMin, RollingMax


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

        assert rv.update is None

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

        assert rv.update is None

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
    

    