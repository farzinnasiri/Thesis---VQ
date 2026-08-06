# Thesis Asset Intake

This directory is for source artifacts used to write thesis chapters. It is separate from `Images/`, which contains final figure files included by LaTeX.

The directory name follows the research-question number, not the chapter
number. The mapping is:

- `rq1/` -> Chapter 4
- `rq2/` -> Chapter 5
- `rq3/` -> Chapter 6
- `rq4/` -> Chapter 7
- `rq5/` -> Chapter 8

Within each research-question directory, use this convention:

- `rq1/data/`: CSV, JSON, or summary tables used for Chapter 4.
- `rq1/figures/`: notebook-exported plots considered for Chapter 4.
- `rq1/reconstruction_samples/`: qualitative reconstruction examples, preferably one folder per sample containing `orig.png`, `vqgan.png`, and `llamagen.png`.
- `rq1/notes/`: short text notes about caveats, run settings, or interpretation.

Only compact artifacts needed for analysis or writing are copied here. The
analysis host keeps the complete `chapterN_outputs/` directories produced by
the notebooks, and `Images/` contains the final figures included by LaTeX.

Keep raw heavyweight experiment outputs outside this thesis repo unless they are small enough and directly needed for writing. Prefer compact tables, selected plots, and representative image samples.
