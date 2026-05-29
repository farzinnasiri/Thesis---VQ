# Thesis Asset Intake

This directory is for source artifacts used to write thesis chapters. It is separate from `Images/`, which contains final figure files included by LaTeX.

Use this convention:

- `rq1/data/`: CSV, JSON, or summary tables used for Chapter 4.
- `rq1/figures/`: notebook-exported plots considered for Chapter 4.
- `rq1/reconstruction_samples/`: qualitative reconstruction examples, preferably one folder per sample containing `orig.png`, `vqgan.png`, and `llamagen.png`.
- `rq1/notes/`: short text notes about caveats, run settings, or interpretation.

For later chapters, use the same pattern under `rq2/`, `rq3/`, `rq4/`, and `rq5/` as needed.

Keep raw heavyweight experiment outputs outside this thesis repo unless they are small enough and directly needed for writing. Prefer compact tables, selected plots, and representative image samples.
