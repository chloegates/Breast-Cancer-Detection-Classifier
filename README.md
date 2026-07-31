# Breast Cancer Diagnosis Prediction Using Random Forest

Final course project: classify breast tumors as **benign** or **malignant** using a tuned **Random Forest** on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Results (locked holdout, n=114)

| Metric | Value |
|--------|------:|
| Accuracy | 0.9737 |
| Precision (malignant) | 1.0000 |
| Recall (malignant) | 0.9286 |
| F1 | 0.9630 |
| MCC | 0.9442 |
| ROC-AUC | 0.9964 |

5-fold CV accuracy (development, n=455): **0.9648 ± 0.0128**

## Quick start

```bash
pip install numpy pandas scikit-learn joblib matplotlib jupyter
python train_random_forest_model.py
```

Full instructions: **[HOW_TO_RUN.md](HOW_TO_RUN.md)**

## Repository contents

- `wdbc_clean.csv`, `development_unscaled.csv`, `test_unscaled_FINAL_HOLDOUT.csv` — data
- `03_Model_Preprocessing.ipynb` — preprocessing / split
- `Random_Trees.ipynb` — RF training notebook
- `train_random_forest_model.py` — reproducible training script
- `models/random_forest_cv5_model.joblib` — trained model
- `random_forest_cv5_results.json` — metrics
- `report_figures/` — EDA and result plots
- `Final_Report_IEEE_Breast_Cancer_RF.md` — final report source

## Method summary

- Algorithm: Random Forest (bootstrap aggregating + `max_features=sqrt` feature randomness)
- Validation: 5-fold stratified CV + GridSearchCV
- Best params: 300 trees, `min_samples_split=5`, `class_weight=balanced_subsample`
- Features: 30 unscaled morphological measurements

## Citation / report

See `Final_Report_IEEE_Breast_Cancer_RF.md` (export to IEEE Word/PDF for course Dropbox).

**GitHub:** replace this README URL after you push:

`https://github.com/<YOUR_USERNAME>/<YOUR_REPO>`

## License

Course project materials — for academic submission use.
