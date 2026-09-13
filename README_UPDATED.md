# ModelGuard - Updated DLRL Version

## Purpose
ModelGuard is a local/offline ML model validation and testing framework.

## Workflow
ML Model + Test Dataset (+ optional Reference/Baseline Dataset)
-> Model Loading
-> Dataset Upload & Validation
-> Data Imputation (only runs if missing values were found)
-> Preprocessing Testing
-> Model Inference
-> Result Validation
-> Robustness Testing
-> Out-of-Distribution Detection
-> Distribution Shift Detection
-> Data Poisoning Detection
-> Model Poisoning Detection (backdoor/trigger indicators)
-> Selenium UI Validation
-> Automated Word Report
-> Final PASS / FAIL

## Main technologies
- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- Scikit-learn
- pytest
- Selenium
- python-docx
- Python logging

## New stages (this update)

**Data Imputation** — missing values no longer hard-fail validation by
default. The Dataset page has a toggle to enable/disable automatic
imputation and a strategy selector (mean / median / most_frequent / knn).
When enabled, a missing-values dataset gets a WARNING (not FAIL) on Dataset
Upload & Validation and is imputed in a dedicated Data Imputation stage that
reports exactly which columns/how many values were changed.

**Out-of-Distribution (OOD) Detection** — flags samples in the test dataset
that look statistically unusual. With a reference dataset supplied, it uses
per-feature z-scores against the reference mean/std. Without one, it falls
back to a weaker self-consistency check (Isolation Forest on the dataset
itself) and says so explicitly.

**Distribution Shift Detection** — compares the test dataset's feature
distributions against a reference/baseline dataset using the
Kolmogorov-Smirnov test and Population Stability Index (PSI) per numeric
feature. This stage is SKIPPED (not PASS/FAIL) if no reference dataset is
supplied, since there is nothing to compare against.

**ML Security Testing (data poisoning indicators)** — extended beyond the
original duplicate/outlier/rare-label checks with a multivariate anomaly
detector (Isolation Forest across all numeric features together) and a
nearest-neighbor label-consistency check (flags samples whose label
disagrees with most of their nearest neighbors, a classic label-flipping
poisoning signature).

**Model Poisoning Detection** — new stage that probes the trained model
itself for backdoor/trigger indicators: single-feature extreme-value
probing (does one feature alone collapse predictions to one class?),
random-noise output entropy (does the model output one class too
confidently on random inputs?), and feature-importance/weight concentration.

All of these are documented, in the code and in the UI, as **indicators for
manual review, not proof** of poisoning or a compromised model — same
caveat as the original ML Security Testing stage.

The Reference/Baseline Dataset upload (Dataset page, optional) powers OOD
Detection and Distribution Shift Detection. It can also be set via the
`REFERENCE_DATASET_PATH` environment variable for local/offline runs. The
imputation strategy can similarly be set via `IMPUTE_STRATEGY`.

## Important
The sample model is a scikit-learn/joblib model. When the DLRL guide provides the real model,
the framework should use that model's actual input features and output/label definition.

Data Poisoning Detection contains basic poisoning indicators (duplicates, extreme outliers,
rare labels, multivariate anomalies and label-neighbour inconsistency). These indicators do not
prove that a dataset is poisoned.

## Run locally
1. Install the packages from requirements.txt in an offline environment where the wheels
   are already available.
2. Start the application:
   py -3.11 -m streamlit run app.py
3. Open the local Streamlit address.
4. Upload the model on Model page.
5. Upload the CSV test dataset on Dataset page.
6. Select the ground-truth label column if the dataset has one.
7. Open Validation and click Run ModelGuard Validation.
8. Download the generated Word report from Reports.
9. For Selenium (offline):
   - Keep the Streamlit app running.
   - Install Chrome and ChromeDriver locally.
   - Set CHROMEDRIVER_PATH to the local driver executable, or place chromedriver.exe in drivers/.
   - The project does not use Selenium Manager/download a driver.
   - Click Run Selenium UI Validation.

## Automated tests
Core tests (no browser):
    py -3.11 -m pytest

Selenium UI test:
    py -3.11 -m pytest -m ui -q

The default pytest configuration excludes UI tests because a browser and running local
Streamlit server are required.

## Files changed
- app.py — Dataset page: imputation controls + reference dataset upload; Validation page wires new params; Dashboard shows new stages
- core/validator.py — data imputation wiring, extended ML Security checks, new check_model_poisoning / check_ood / check_distribution_shift, extended run_validation_pipeline
- core/imputation.py (new) — mean/median/most_frequent/knn imputation
- core/config.py — reference_dataset_path() / impute_strategy() env helpers
- tests/test_model_integrity.py
- tests/test_dataset.py
- tests/test_data_validation.py
- tests/test_preprocessing.py
- tests/test_inference.py
- tests/test_results.py
- tests/test_robustness.py
- tests/test_ml_security.py
- tests/test_ui_selenium.py
- tests/test_report.py
- tests/test_data_integrity.py
- tests/test_imputation.py (new)
- tests/test_model_poisoning.py (new)
- tests/test_ood_distribution_shift.py (new)
- reports/report_generator.py
- utils/logger.py
- model/create_model.py
- dataset/create_dataset.py
- pytest.ini
- requirements.txt (added scipy)

## Run the new checks locally
1. On the Dataset page, upload your test dataset. Optionally toggle
   automatic imputation and pick a strategy. Optionally upload a
   reference/baseline dataset for OOD + distribution shift checks.
2. On the Validation page, click Run ModelGuard Validation as before —
   the new stages run automatically as part of the same pipeline.
3. Automated tests: `py -3.11 -m pytest` now also runs
   test_imputation.py, test_model_poisoning.py, and
   test_ood_distribution_shift.py.


## Feature status verified in the supplied project
- Data Imputation: implemented with mean, median, most_frequent and KNN strategies; imputation is reported as WARNING when applied rather than silently treated as PASS.
- Data Poisoning Detection: implemented using duplicate checks, extreme outliers, rare labels, Isolation Forest anomalies and nearest-neighbour label consistency.
- Model Poisoning Detection: implemented using extreme single-feature trigger probing, random-input dominance checks and feature-importance concentration when exposed by the model.
- Selenium UI Validation: implemented for the localhost workflow. It is configured for offline operation and requires a locally installed ChromeDriver.
- OOD and Distribution Shift: implemented; a real reference/baseline dataset is recommended for meaningful comparison.

## Important limitation
The current bundled sample model is a scikit-learn/joblib model. The DLRL deep-learning model must be integrated after its framework, file format, input shape and output/label definition are confirmed. Do not treat the current poisoning checks as a formal proof of absence of poisoning or a backdoor.


## Automated test count
The current project contains 34 pytest test functions: 33 non-browser tests and 1 Selenium UI end-to-end test. The default `pytest` command excludes the UI test because it requires a running local Streamlit server and a locally installed ChromeDriver.
