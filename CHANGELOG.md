# Changelog

## 1.3.0 - 2026-09-14

Clean audited baseline for regeneration of publication results.

- Fixed headerless HTRU2 loading and feature names.
- Moved training-fraction sampling inside each outer training split.
- Added nested deterministic class-stratified fraction sampling.
- Added cross-fitted target-order encoding for training rows.
- Standardized the fold-contained representation before distance-based generation.
- Corrected SMOTE-style interpolation to use one scalar gap per row.
- Redefined minority-only regimes relative to the real minority count.
- Added corrected repeated-cross-validation inference and broad Holm families.
- Added matched holdout comparators and paired stratified bootstrap intervals.
- Added data hashes and environment metadata to run manifests.
- Replaced print-only Holm checks with automated tests.
- Removed hard-coded legacy figure values from the clean package.
