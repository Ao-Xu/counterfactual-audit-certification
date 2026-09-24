# Matched pairing and anchor-reuse study

This controlled factorial compares separate versus paired evaluation and
split versus shared anchors. The frozen confirmation phase uses eight design
cells, 13 methods, 125 audit budgets, 1,600 replications, and the seed and
decision rules in `frozen_manifest.json`.

From a fresh `code/` copy, after installing `requirements.txt`:

```powershell
python -u -m pairing_analysis.simulate confirm
python -u -m pairing_analysis.forecast
python scripts/export_attribution.py --output-dir ../pairing_export
```

The commands require their frozen configuration and generate new result files;
they do not read result tables bundled with this repository. Some later
analysis commands additionally require earlier stress or ReCoRD outputs that
are **not** supplied here. The export command is for the matched factorial,
not a claim that every historical appendix diagnostic has been reproduced.

The 95% bands used for threshold inversion are simultaneous across 13 methods
and 125 budgets **within each geometry**, not across all eight geometries.
Missing first-grid crossings are censored, not zero-cost observations.
