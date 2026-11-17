# Finishing plan

## 1. Treat all legacy numerical results as provisional

The uploaded legacy result files were generated before the current audit was complete. At least the following implementation or inference changes can alter the reported values:

1. HTRU2 is now read without losing the first headerless observation.
2. Training fractions are selected inside each outer training fold, while validation rows stay fixed across fractions.
3. target-order training encodings are cross-fitted.
4. distance-based interpolation operates on fold-standardized data.
5. interpolation uses one scalar gap per synthetic row, consistent with the line-segment formula in the manuscript.
6. minority-only sample counts are based on the real minority count rather than the total training-fold size.
7. repeated-cross-validation inference uses a corrected standard error.
8. Holm adjustment covers the full searched comparison family rather than one generator/fraction at a time.
9. holdout evaluation uses a same-fraction Real-only comparator and paired bootstrap intervals.

Therefore, do not manually patch the old manuscript tables. Regenerate the experiment outputs and rebuild the Results section from those outputs.

## 2. Push the clean code baseline

Before publishing the repository:

1. Add an explicit license selected by the authors.
2. Run `python -m pytest`.
3. Run the smoke test from the root README.
4. Confirm that no raw CSV files, virtual environments, or legacy result folders are staged by Git.
5. Commit the baseline before running expensive experiments, for example:

```text
Audit and clean reproducible experiment pipeline
```

Tagging this baseline, for example `code-audit-v1`, will make the final result provenance easier to explain.

## 3. Run experiments in this order

### A. Main logistic-regression grid

Run all four datasets, all three generators, all four training fractions, and the class-conditional regimes. Inspect:

- no failed or missing split scores;
- exactly 50 split scores for every configuration and metric;
- identical Real-only split scores across generator loops;
- correct sample counts and class proportions;
- sensible corrected confidence intervals;
- Holm family sizes that include all searched generators, regimes, fractions, and metrics for each dataset/model analysis.

### B. HTRU2 check

HTRU2 is already part of the main grid. Confirm 17,898 rows and a positive rate of approximately 9.16% in `run_manifest.json`. Do not add HTRU2 to the paper's Abstract or Results until its complete corrected grid is available.

### C. Random-forest robustness grid

Run this only after the logistic-regression output passes the checks above. Random forest will be the most expensive part of the present plan.

### D. Minority-only sensitivity

Use the new regime names and definitions. Check whether the paper still needs all three generator families here. If time is tight, the scientifically central comparison is Real-only versus SMOTE-style minority replacement/augmentation, with bootstrap as a conservative control.

### E. Ordinal-encoding sensitivity

Run only Adult and Default Credit. Numeric-only datasets do not provide an encoding sensitivity test.

### F. Freeze development selection, then evaluate holdout once

Do not run the final holdout script until the development grid, model list, generators, fractions, regimes, preprocessing choices, and selection rule are final. Save the row-level prediction file because it is required for the paired bootstrap intervals.

## 4. Decide whether an additional method or dataset is necessary

Do not add SMOTENC, Online Shoppers, CTGAN, or another large experiment before seeing the corrected four-dataset results.

A focused decision rule:

- If Adult and Default Credit remain difficult to interpret because ordinary interpolation operates on encoded categorical geometry, add SMOTENC as one minority-only categorical-aware sensitivity baseline and then add Online Shoppers as a clean mixed-type test.
- If the corrected four-dataset results already produce a coherent conditional finding, finish the paper without expanding the scope.
- Add CTGAN only if a target journal or reviewer requires a modern neural baseline. It is lower priority than correct inference, matched holdout comparison, and clear figures.

## 5. Build final figures after the grid is frozen

Recommended main-paper figures:

1. Experimental workflow diagram.
2. Corrected paired-effect forest plot for the main logistic-regression comparisons.
3. Training-fraction learning curves or delta curves showing when effects appear.
4. Compact matched holdout comparison with paired bootstrap intervals.

Keep exact full grids, encoding sensitivity, minority-only detail, and most corrected p-values in Supporting Information.

## 6. Rewrite the paper from outputs rather than preserving old prose

After all reruns:

1. regenerate tables and draft plots from CSV files;
2. replace every hard-coded numerical statement;
3. write Results descriptively from the corrected effect sizes and intervals;
4. rewrite Discussion claims as explanations consistent with the evidence, not established causal mechanisms;
5. complete a separate language edit only after numbers and figures are fixed.

## 7. Final reproducibility release

The final repository or tagged release should contain:

- source code and tests;
- environment definition and lock file generated on the final machine;
- data acquisition instructions, not raw restricted data;
- final run manifests with checksums;
- fold-level or archived result files;
- final statistical comparison files;
- row-level holdout predictions if sharing is permitted;
- figure-generation scripts;
- a versioned manuscript source or a link to the publication archive.
