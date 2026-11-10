# 2. Treating every column as a category

Trees split numbers by thresholds (`Age < 39.5`), assuming nearby values behave alike. In synthetic data they often
don't: each exact flight distance corresponds to a route, and routes differ.

CatBoost turns each category into an **ordered target statistic**: the satisfaction rate of *earlier* rows with the
same value, plus a prior. A row's own label never enters its own feature, which avoids the leakage plain target
encoding causes. Declaring every column (even `Age`, `Flight Distance`, the delays) as a category gives each exact value
its own learned rate. This lifted the blend by 0.0004 (exp10), and the blend gave it weight 0.50.

It adds to the lookups in note 1 because those learn rates from the *real* data and this learns them from the
*synthetic* labels. They disagree where the generator changed a pattern, which a blend can use. The neural network
version (one embedding per exact value) added +0.00013.

**Check:** would this help with only 1,000 rows? *Probably not: too few rows per value. With 700k rows there are
hundreds per value.*
