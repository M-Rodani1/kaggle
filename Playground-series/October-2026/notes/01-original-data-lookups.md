# 1. The original dataset as a lookup table

Playground data is synthetic: a generator learned a real survey (130k passengers) and produced new rows. The real data
still holds the cleanest version of the patterns.

- **Appending the original rows to training lost 0.0002.** Real and synthetic rows follow slightly different rules.
- **Turning it into features gained 0.0004.** For each value, take the satisfaction rate in the real data:

```text
rate(value) = (satisfied with that value + 20 × overall rate) / (rows with that value + 20)
```

The 20 is a pseudo-count that pulls rarely seen values toward the overall rate. Keys used: each single column, the full
profile of 13 ratings + 4 categories, and the 4 categories alone.

**Why it can't leak:** the rates use only the original dataset's labels, never the competition's, so they can be
computed once and used in every fold.

**Check:** why does a profile seen once in the original, with a satisfied passenger, get a rate near 0.46 and not 1.0?
*The pseudo-count dominates a single row: (1 + 20 × 0.43) / 21 ≈ 0.46.*
