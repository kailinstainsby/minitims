# Purpose and Scope

## Purpose

This project is a small, compact python library for incremental rolling-window statistics for time-series.

The purpose is to impliment a minimal and well-tested library that provides incremental rolling-window statistical primitives for univariate time series data.

The library focuses on window semantics and stateful aggregation, enabling statistics such as mean, variance, and extrema to be update efficiently as new observations arrive, without recomputing over the entire window.

## Scope

- Works on 1D numeric time series
- Supports fixed length rolling windows
- Computes basic statistics incrementally 
- Exposes a clean, explicit API
- is fully unit tested
- is easy to reason about

it will NOT:

- do forecasting / prediction
- handle multivariate series
- use pandas
- handle missing timestamps
- optimise with numpy vectorisation
- impliment ML models
- include plotting or visual
- claim performance parity with production libraries

again, this is a small, compact library for specifically univartiate primitives, built for the sole purpose of feeding into and teaching me about my next project, which is a real-time streaming analytics pipeline with stateful computation, windowing logic, which guarentees correctness. it does not compete with pandas, aeon, etc.

## Questions to answer

- What happens when the window is not full?
- What happens when a new value arrives and the window is full?
- What does “mean” mean in this system?
- When can the system return a value?
- What must always be true after update()?
