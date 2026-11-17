# Data files

Raw datasets are not stored in this repository. Place the following files in this directory or pass another directory through `--data-dir`:

```text
cs-training.csv
UCI_Credit_Card.csv
adult.csv
HTRU_2.csv
```

## Give Me Some Credit

- Expected target: `SeriousDlqin2yrs`
- The loader drops `Unnamed: 0` when present.
- Expected local file: the 150,000-row Kaggle training CSV.

## Default of Credit Card Clients

- Expected target: `default.payment.next.month`
- The loader drops `ID` when present.
- `SEX`, `EDUCATION`, `MARRIAGE`, and `PAY_0`, `PAY_2` through `PAY_6` are routed through categorical processing.

## Adult Census Income

- Expected target: `income`
- The project CSV uses column names such as `marital.status`, `education.num`, and `native.country`.
- Question-mark entries are treated as missing.

## HTRU2

- Expected file: the original headerless 17,898-row CSV.
- Column order used by the loader:
  1. mean of the integrated profile;
  2. standard deviation of the integrated profile;
  3. excess kurtosis of the integrated profile;
  4. skewness of the integrated profile;
  5. mean of the DM-SNR curve;
  6. standard deviation of the DM-SNR curve;
  7. excess kurtosis of the DM-SNR curve;
  8. skewness of the DM-SNR curve;
  9. class.

The loader is tolerant of a locally added header, but the raw headerless file is preferred so its checksum remains comparable with the source file.
