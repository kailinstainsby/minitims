# minitims

**Incremental rolling-window statistics for time-series data**

A lightweight Python library providing stateful, O(1) rolling statistical primitives for streaming data. Built for signal processing pipelines, quantitative systems, and real-time analytics where data arrives one observation at a time.

---

## DISCLAIMER !!
this README was written with help from Claude Sonnet 4.5. Rest assured it was not entirely generated, only assisted, and all has been fact checked and verified. The code itself was written by hand with minimal assistance, and the logic is all designed by me. Enjoy!

---

## What is this?

`minitims` implements fixed-window statistical aggregators that update incrementally as new data arrives, without recomputing over the entire window. This is the foundation layer for building streaming signal engines, feature pipelines, and real-time analytics systems.

**What it does:**
- Maintains rolling windows of configurable size
- Updates statistics in constant time O(1) or O(n) depending on operation
- Uses bounded memory (fixed-size buffers, no growing arrays)
- Provides clean, explicit API with well-defined semantics

**What it doesn't do:**
- Forecasting or prediction
- Handle multivariate series
- Include pandas integration
- Provide plotting/visualization
- Claim performance parity with production libraries like NumPy

This is a focused, well-tested library for **univariate rolling primitives**.

---

## Installation

```bash
pip install -e .
```

---

## Quick Start

```python
from minitims import RollingMean, RollingStdDev, RollingReturns

# Rolling mean with window size 5
rm = RollingMean(window_size=5)

for price in [100, 102, 101, 105, 103]:
    rm.update(price)
    if rm.is_full():
        print(f"Mean: {rm.get_mean()}")

# Log returns from price stream
rr = RollingReturns(window_size=10)
prices = [100, 105, 103, 108, 110]

for price in prices:
    rr.update(price)
    if rr.get_current_return():
        print(f"Return: {rr.get_current_return():.4f}")
```

---

## Features

### Core Primitives

- **`RollingMean`** — O(1) incremental mean via running sum
- **`RollingVariance`** — O(1) incremental variance via sum of squares
- **`RollingStdDev`** — O(1) standard deviation (derived from variance)
- **`RollingMin` / `RollingMax`** — O(n) extrema tracking
- **`RollingReturns`** — Log returns from price stream
- **`RollingZScore`** — Standardized values relative to rolling window

### Window Semantics

- Fixed-size windows (defined at initialization)
- Oldest value evicted when window is full
- Methods return `None` until window is full (explicit handling)
- `get_window()` returns current buffer state as list

---

## Design Principles

1. **Incremental updates** — No recomputation on each tick
2. **Bounded memory** — Fixed-size buffers, no growing arrays
3. **Explicit API** — Clear semantics for partial windows vs full windows
4. **Well-tested** — Comprehensive unit tests with edge cases
5. **Composable** — Build complex signals from simple primitives

---

## API Overview

All rolling classes inherit from `RollingWindow` and share:

```python
# Common interface
.update(value)        # Add new value to window
.is_full()            # Check if window has reached size
.get_window()         # Get current buffer as list
```

### RollingMean
```python
rm = RollingMean(window_size=10)
rm.update(42)
mean = rm.get_mean()  # Returns None until full
```

### RollingReturns
```python
# Feed PRICES, stores RETURNS
rr = RollingReturns(window_size=20)
rr.update(100.5)  # Price
rr.update(102.3)  # Price
ret = rr.get_current_return()  # Most recent return
avg = rr.get_mean_return()     # Average return (when full)
```

### RollingZScore
```python
rz = RollingZScore(window_size=50)
rz.update(value)
zscore = rz.get_zscore()        # Z-score of most recent value
zscore = rz.get_zscore(105.0)   # Z-score of arbitrary value
```

---

## Testing

```bash
pytest tests/
```

All classes have comprehensive test coverage including:
- Initialization (empty, partial, full, oversized)
- Update behavior (growth, eviction, invariants)
- Result calculation (correctness, edge cases)

---

## Use Cases

- **Signal processing pipelines** — Normalize, filter, and transform streaming data
- **Feature engineering** — Generate rolling stats for ML models
- **Quantitative research** — Build alpha signals from market data
- **Real-time monitoring** — Track metrics over sliding windows

---

## Why Not Just Use NumPy/Pandas?

You should! For batch processing, vectorized operations, and exploratory analysis, use the full-featured libraries.

`minitims` is for when you need:
- **Streaming updates** (one observation at a time)
- **Stateful computation** (maintain window across calls)
- **Explicit control** over window semantics
- **Minimal dependencies** for embedded/constrained environments

---

## License

MIT

---

## Roadmap

Future extensions (not currently implemented):
- Exponential moving averages (EMA)
- Weighted statistics
- Multi-asset support
- Serialization/deserialization
