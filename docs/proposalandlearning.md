What you still need to read (short, targeted list)

You do not need:

Metaclasses

abc.ABC

Advanced OOP patterns

NumPy internals

Ignore those for now.

You do need these 5 things:
(A) Mutable state & invariants (you’re halfway)

You already started this. Key idea:

Some variables must always be consistent with each other.

Example invariant:

running_sum == sum(buffer)


That’s it.

(B) Ring buffers / queues (very important)

Read about:

FIFO queues

Fixed-size buffers

What happens when capacity is exceeded

This is the mechanical heart of the project.

(C) Incremental algorithms (tiny exposure)

Just understand:

Why recomputing everything is bad

Why adding/removing elements matters

You don’t need theory — just intuition.

(D) Partial results (edge cases)

Understand:

“Not enough data yet”

Why returning None or raising is a design choice

This is core to time series infra.

(E) Testing stateful systems

This is subtle:

Order matters

Tests must simulate sequences

This is different from testing pure functions.

That’s the entire prerequisite list.