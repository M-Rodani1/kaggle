# 4. Leaderboard noise and final picks

The public leaderboard scores about 20% of the test rows; the private one scores the other 80% and decides the ranking.
A public score is a noisy sample of the real one.

**Noise rule of thumb:** the public noise in the difference between two files ≈ 0.003 × the standard deviation of the
difference of their percentile ranks. Near-identical files can't be separated; genuinely different models can be
separated by about 0.0001.

**Tuning on the public board is fitting it.** Submit many variants, keep the best, and the winner's lead is part real
and part luck that won't repeat on the other 80%.

**Final picks.** Kaggle counts the better of your two picks on the private board, so the second pick can only help.

| Pick | Public | Private |
| --- | ---: | ---: |
| exp16 stack, validated by nested CV | 0.96186 | **0.96239** |
| Public blend tuned on the leaderboard | 0.96213 | 0.96222 |

The blend led by 0.00027 on public and trailed by 0.00017 on private: the winner's curse in practice. The validated
stack counted, and finished 38th.

**Check:** two files score 0.96186 and 0.96189 with rank correlation 0.99999. Which? *Either: the difference is far
inside the noise.*
