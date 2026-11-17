# Manuscript audit before revision

No manuscript text has been edited in this clean repository. The current LaTeX draft should be preserved as a legacy snapshot until the corrected experiments are complete.

## Highest-priority scientific revisions

### 1. Replace every legacy result table and numerical claim

The current tables were produced under an earlier pipeline and statistical procedure. They should not be repaired manually. The corrected code changes the training-fraction design, target-order encoding, interpolation geometry, minority-only regimes, repeated-CV uncertainty, Holm families, and holdout comparison.

### 2. Complete the statistical-methods subsection

The manuscript currently contains commented drafting notes rather than a finished procedure. The final subsection should explain:

- matched split-level differences;
- corrected repeated k-fold t inference for development comparisons;
- the exact Holm family;
- effect sizes and corrected confidence intervals as primary summaries;
- paired class-stratified percentile bootstrap intervals for final holdout differences;
- no ordinary bootstrap p-value unless a separate valid null test is added.

### 3. Reconcile the dataset count

The Methods table includes HTRU2, while the Abstract, Introduction, several limitations, and the Results narrative still describe three datasets. HTRU2 should appear throughout only after its full corrected experiments are complete.

### 4. Rewrite the training-fraction description

The corrected design defines repeated cross-validation on the full development set and selects the requested fraction only from the training side of each split. Validation rows remain fixed across fractions. The final selected model uses the selected fraction of the development set, not automatically the full development set.

### 5. Rename and redefine minority-only regimes

The legacy label "Synthetic-only" is inaccurate when real majority-class observations remain in the training set. Use the explicit terms:

- Minority-replaced;
- Minority +1x;
- Minority +3x.

These amounts are now relative to the original minority count. Do not describe Minority-replaced as replacement of the complete real dataset.

### 6. Describe target-order encoding precisely

The revised code cross-fits target-order encoding for outer-training rows and fits the final mapping on the complete outer training fold for validation and holdout transformation. The manuscript should describe this distinction. It should also acknowledge that encoded categorical geometry remains a substantive modeling choice.

### 7. Describe standardization before interpolation

The nearest-neighbor search now operates on a standardized fold-contained numeric representation. This should be explicit because distance calculations on the unscaled financial and census variables would otherwise be dominated by large-magnitude columns.

### 8. Align the interpolation formula and implementation

The final implementation uses a single scalar interpolation coefficient per synthetic row, so the row lies on the line segment between a seed and one same-class neighbor. The manuscript formula can then remain in its current scalar form.

### 9. Correct model descriptions

The final paper should list all explicit model settings. In particular:

- logistic regression uses balanced class weights, `lbfgs`, `max_iter=2000`, and `C=1`;
- random forest uses 300 trees, `min_samples_leaf=2`, and `balanced_subsample` class weighting.

The number of trees improves ensemble stability; it is not what gives trees their nonlinear or interaction capacity.

### 10. Rebuild holdout tables

Every selected synthetic procedure must be compared with Real-only at the same real-data fraction on the same holdout observations. The final table should report:

- dataset and model;
- development selection metric;
- selected procedure;
- matched Real-only procedure;
- selected and Real-only holdout scores;
- paired delta;
- paired bootstrap 95% interval.

Do not compare development means directly with holdout scores as evidence of improvement or lack of overfitting.

## Interpretation and framing

### Dataset inclusion

The "Why included" column should use properties known before observing performance. Remove outcome-dependent phrases such as "clearest preliminary positive case."

### Causal language

Dataset size, class imbalance, numerical feature structure, local smoothing, and model family are plausible explanations, but the current design does not isolate them. Present these as hypotheses consistent with the pattern.

### Bootstrap language

Bootstrap resampling reproduces the empirical training distribution by reweighting existing rows. It does not transform every feature toward a new parametric distribution.

### Numerical versus continuous

Give Me Some Credit has no nominal categorical predictors, but several variables are discrete counts. Use "predominantly numerical" or "no nominal categorical predictors," not "entirely continuous."

### Scope

The strongest defensible contribution is a careful empirical framework showing that utility is conditional on dataset, generator, regime, preprocessing, model, and real-data budget. The paper does not need a universal positive effect.

## Structural and language revisions

After the final outputs are frozen:

1. remove all internal drafting comments and instructions;
2. rewrite the Abstract from the final four-dataset results;
3. shorten Results and replace most tables with a small number of figures;
4. move full numerical grids to Supporting Information;
5. rewrite Discussion in a more cautious, evidence-linked form;
6. replace the opening sentence "The project should not claim..." in Conclusions with a direct scientific conclusion;
7. perform a full grammar and consistency edit, including dataset names, capitalization, hyphenation, percentages, `holdout` versus `hold-out`, and `Real-only` regime names.

## Recommended scope decision

The current four-dataset design may be sufficient once it is corrected and fully rerun. SMOTENC plus Online Shoppers is the most defensible optional extension if categorical interpolation remains a central reviewer concern. CTGAN is optional and should not delay completion unless a target venue explicitly expects a modern neural generator.
