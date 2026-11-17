# GitHub upload checklist

- [ ] Add a software license agreed upon by the authors.
- [ ] Confirm author names and repository ownership.
- [ ] Run `python -m pytest`.
- [ ] Run the smoke test.
- [ ] Confirm `git status` does not show `.venv`, raw CSV files, or legacy result folders.
- [ ] Review `README.md`, especially data filenames and experiment commands.
- [ ] Initialize Git and make the baseline commit.
- [ ] Push the baseline before running expensive experiments.
- [ ] Generate and commit `uv.lock` on the machine used for the final runs, if uv is the chosen environment manager.
- [ ] Do not commit old manuscript tables as current results.
- [ ] After final reruns, add only the curated outputs needed for reproducibility or place large outputs in a tagged release.
