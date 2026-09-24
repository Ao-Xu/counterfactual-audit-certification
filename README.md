# Counterfactual audit certification

Research code for **“When Can Counterfactual Audits Certify a Fine-Tuning Gain?”**

Start with the [code guide](code/README.md) or download [code.zip](code.zip).
Both contain the same source tree. The release is organized by the role of each
experiment, not by internal run names or chronological versions.

| Code area | Purpose |
|---|---|
| [`planning`](code/planning/) | Independent-pilot audit planning and final-power study |
| [`pairing_analysis`](code/pairing_analysis/) | Matched pairing × anchor-reuse factorial |
| [`certificate_efficiency`](code/certificate_efficiency/), [`sampled_interventions`](code/sampled_interventions/), [`shared_anchor`](code/shared_anchor/) | Controlled certificate comparisons and allocation checks |
| [`finite_catalog`](code/finite_catalog/), [`identification_audit`](code/identification_audit/) | Finite-catalog selection and audit diagnostics |
| [`language_models`](code/language_models/) | Qwen/MultiNLI, HANS/ANLI, and related text studies |
| [`scripts`](code/scripts/) | Result exports, interval checks, and figure scripts |

The public archive contains source, frozen configurations, and tests. It does
**not** include model weights, licensed data, raw predictions, saved result
tables, or the manuscript's figure PDFs. This is therefore not a one-command
reproduction of every figure. See the [experiment map](code/EXPERIMENTS.md)
for the required inputs and the exact boundary of each workflow.

No open-source license has been selected. Please contact the authors before
reuse beyond what copyright law permits.
