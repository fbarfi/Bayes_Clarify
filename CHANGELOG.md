# Changelog

## 0.1.0 (2026-11-07)

First release.

* `Posterior` from any posterior draws and a prediction rule; `from_linear` for
  GLM-type models; `from_pymc` to predict through a PyMC model.
* Draws from ArviZ InferenceData, CmdStanPy, NumPyro, and draws tables (CSV or
  DataFrame, e.g. exported from R/brms or Stan).
* `sim_apply`, `sim_params`, `sim_setx`, `sim_ame`, `sim_predictions`
  (`sim_adrf` as an alias), following the R package clarify.
* `Draws.summary` (equal-tailed or highest-density intervals, posterior
  probabilities of direction), `transform`, `concat`, `plot`, `to_latex`.
* Predictions are averaged over units chunk by chunk of draws, so large panels
  do not need the unit-level predictions of all draws in memory.
