# Concept 4: Blending models by hill climbing

A blend helps when models make *different* mistakes. LightGBM, CatBoost, and XGBoost are all gradient-boosted trees, but each builds trees differently:

| Model | How it grows trees | How it handles categories |
| --- | --- | --- |
| LightGBM | Leaf-wise: splits the most useful leaf next | Groups categories at each split |
| CatBoost | Symmetric: the same split across a whole tree level | Its own ordered target statistics |
| XGBoost | Depth-wise: level by level, up to `max_depth` | Native categories (`enable_categorical`) |

When one model ranks a buyer too low, another often ranks them correctly. Averaging their probabilities cancels part of each model's error.

## Choosing the weights

Before exp07, the notebook tried 21 LightGBM/CatBoost weights (0.00, 0.05, …, 1.00) and kept the best. With three or more models, a grid grows quickly. **Hill climbing**, also called *ensemble selection*, scales better:

```text
1. Start with the best single model.
2. Try adding one more "vote" for each model and score each candidate blend on OOF.
3. Keep the vote that scores best, even if it doesn't improve (this lets the search escape small dips).
4. Repeat 50 times, then use the best blend seen along the way.
```

A model's weight is its share of the votes. For example, 25 CatBoost, 15 LightGBM, and 10 XGBoost votes give weights of 0.50, 0.30, and 0.20. A model that never helps gets no votes and a weight of 0.

## A caution

The weights are chosen on the same OOF predictions we then report, so the blend's OOF score is slightly optimistic. With three models and only 50 votes the effect is tiny. It would grow if we blended dozens of models, and that's when to confirm the weights on a second fold seed (Concept 1).

**Check your understanding:** If XGBoost scores lower than CatBoost on its own, can it still get a weight above 0? **Yes. What matters is whether it adds information the other models lack, not whether it beats them.**
