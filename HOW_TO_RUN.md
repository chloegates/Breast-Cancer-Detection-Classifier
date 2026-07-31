# How to Run the Code

**Project:** Breast Cancer Diagnosis Prediction Using Random Forest  
**Course final submission companion document**

---

## 1. Requirements

- Python 3.9+ (Anaconda/Miniconda recommended)
- Packages:
  - `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `jupyter` (for notebooks)

Install (example):

```bash
pip install numpy pandas scikit-learn joblib matplotlib jupyter
```

---

## 2. Project layout (key files)

```
Project Random Forrest/
├── wdbc_clean.csv
├── development_unscaled.csv
├── test_unscaled_FINAL_HOLDOUT.csv
├── 03_Model_Preprocessing.ipynb
├── Random_Trees.ipynb
├── train_random_forest_model.py
├── models/random_forest_cv5_model.joblib
├── random_forest_cv5_results.json
├── random_forest_feature_importance.csv
├── report_figures/          # EDA + results plots
├── Final_Report_IEEE_Breast_Cancer_RF.md / .docx
├── README.md
└── HOW_TO_RUN.md            # this file
```

---

## 3. Option A — Run the training script (recommended)

From the project directory:

```bash
cd "/path/to/Project Random Forrest"
python train_random_forest_model.py
```

This will:

1. Load development + holdout CSVs  
2. Run GridSearchCV with 5-fold stratified CV  
3. Save:
   - `models/random_forest_cv5_model.joblib`
   - `random_forest_cv5_results.json`
   - `random_forest_feature_importance.csv`

**Note:** Grid search can take several minutes.

---

## 4. Option B — Run notebooks

```bash
jupyter notebook
```

1. Open `03_Model_Preprocessing.ipynb` — review cleaning / split logic.  
2. Open `Random_Trees.ipynb` — run all cells top-to-bottom to retrain/evaluate the Random Forest and print metrics.

---

## 5. Using the saved model

```python
import joblib
import pandas as pd

model = joblib.load("models/random_forest_cv5_model.joblib")
test = pd.read_csv("test_unscaled_FINAL_HOLDOUT.csv")
X = test.drop(columns=["id", "diagnosis"])
y = test["diagnosis"].astype(int)

print(model.score(X, y))
```

---

## 6. Regenerating report figures (optional)

Figures used in the final report live in `report_figures/`. They can be regenerated with a short Python plotting script using the saved model + CSVs (matplotlib + scikit-learn metrics displays).

---

## 7. Expected holdout results (reference)

If data/split files are unchanged, you should obtain approximately:

- Accuracy ≈ **0.9737**
- F1 ≈ **0.963**
- MCC ≈ **0.944**
- ROC-AUC ≈ **0.996**

---

## 8. Submission checklist

- [ ] Final IEEE report (Word or PDF)  
- [ ] This `HOW_TO_RUN.md` (or include in README)  
- [ ] Code / notebooks + CSV data needed to reproduce  
- [ ] GitHub link filled in report + README  
- [ ] Confidential peer-evaluation form (separate, from instructor)  
- [ ] Each teammate submits report separately to Dropbox  
