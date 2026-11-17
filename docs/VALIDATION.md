# Validation performed for this clean baseline

The following checks were completed before packaging:

- editable package installation succeeded in the working environment;
- the `synthrep` command-line entry point loaded successfully;
- all 15 automated tests passed;
- a four-dataset smoke run completed with two folds, one repeat, three generators, and one training fraction;
- the smoke run produced fold scores, descriptive summaries, corrected paired tests, and a run manifest;
- the matched holdout script completed on the smoke output and produced selected configurations, paired holdout comparisons, and compressed row-level predictions;
- the data-driven plot and LaTeX-table scripts completed on smoke output.

No full publication experiment was run during this packaging step. The old uploaded result files should not be interpreted as validation of the corrected pipeline because they were produced before several implementation and statistical corrections.
