"""Run notebooks/airline-baseline.ipynb's code cells as one script, with CFG overrides, without editing the notebook.

Usage: python3 scripts/run_notebook.py EXPERIMENT='"exp15_route_aux"' FE_ROUTE=True REUSE='{"CatBoost": "exp10_catboost_cat"}'
Each override is NAME=<Python expression>, applied right after the CFG class (before the DEBUG and FAST adjustments).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
cells = ["".join(c["source"]) for c in json.load(open(ROOT / "notebooks" / "airline-baseline.ipynb"))["cells"]
         if c["cell_type"] == "code"]
overrides = "\n".join(f"CFG.{name} = {value}" for name, value in (a.split("=", 1) for a in sys.argv[1:]))
marker = "\n\nif CFG.FE_INTERACT:"
assert cells[0].count(marker) == 1, "CFG cell layout changed"
cells[0] = cells[0].replace(marker, "\n\n" + overrides + marker)
print("Overrides:\n" + overrides, flush=True)
exec(compile("\n\n".join(cells), "airline-baseline.ipynb", "exec"), {"__name__": "__main__"})
