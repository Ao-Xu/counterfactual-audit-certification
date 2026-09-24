# Independent-pilot planning experiment

This is the frozen controlled experiment for the paper's pilot--plan--certify
analysis. It evaluates three laws (A--C), two target powers (80% and 90%),
and 24 independent pilots per law/target: 144 pilot plans in total. Each pilot
uses 2,048 anchors, split into 256 design and 1,792 estimation anchors.
The manifest fixes candidate budgets, seeds, safeguards, and tie-breakers.

From the `code/` directory, in a fresh working copy:

```powershell
python -m unittest discover -s planning/tests -v
python -u -m planning.run pilot
python -u -m planning.run final
python -u -m planning.run benchmark
python -u -m planning.report
```

Run stages in this order and stop on any nonzero exit. Do not use Python's
`-O` option: the frozen runner uses assertions for integrity checks. Existing
`results/` locks are deliberately not overwritten. To repeat the full study,
use another fresh source copy rather than deleting an audit record.

Outputs are written under `planning/results/`: a locked pilot decision file,
fresh final outcomes, benchmark outcomes, a summary, and generated tables and
figures. This public source archive does not include those generated outputs.

The `manifest.json` design parameters are those of the frozen study. Its
source-byte lock and adjacent checksum were refreshed solely because package
imports were renamed for this public release; the sampling law, budget menu,
seeds, and mathematical estimands were not revised. `freeze.py` creates a new
manifest and is **not** a step in reproducing the fixed run.
