# Breast Cancer Diagnosis Prediction Using Random Forest Classification

**Chloe Gates**  
Georgia State University  
Department of Computer Science  
Atlanta, GA, USA  
Email: chloegates152910@gmail.com

*Note: Replace author block with all team members’ names/emails before final Dropbox submission. Each teammate must submit the report separately.*

---

## Abstract

Breast cancer remains one of the most common cancers among women worldwide, and early differentiation of benign from malignant tumors is critical for clinical decision-making. This paper presents a Random Forest classifier trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset to predict tumor malignancy from thirty numerical features extracted from digitized fine-needle aspirate images of cell nuclei. We preprocess and encode the data, perform an 80/20 stratified split into a development set (455 samples) and a locked holdout set (114 samples), and leave features unscaled because tree-based models are invariant to monotonic feature scaling. Model selection uses GridSearchCV with five-fold stratified cross-validation. The best configuration employs 300 trees, \(\sqrt{p}\) feature sampling at each split, bootstrap aggregating, and balanced subsample class weighting. On development data, mean cross-validated accuracy reaches 96.48% \(\pm\) 1.28%. On the untouched holdout set, the model achieves 97.37% accuracy, F1-score 0.963, Matthews correlation coefficient 0.944, and ROC-AUC 0.996, with zero false positives and three false negatives. Feature importance analysis highlights worst-area, worst-perimeter, and concave-points measurements as dominant predictors. We discuss metric interpretations, limitations regarding false negatives, and future directions including cost-sensitive learning, stronger baselines, and SHAP-based explainability.

**Keywords:** Random Forest, breast cancer diagnosis, ensemble learning, cross-validation, WDBC, machine learning

---

## I. Introduction

Accurate, early classification of breast tumors as benign or malignant can reduce diagnostic delay and support triage decisions after biopsy imaging analysis. Classical clinical assessment of fine-needle aspirate (FNA) samples relies on morphological inspection of cell nuclei; the Wisconsin Diagnostic Breast Cancer (WDBC) dataset encodes those morphological properties as quantitative features suitable for supervised learning [1], [2].

Machine learning models can learn nonlinear decision boundaries over such features and provide reproducible predictions. Among tabular classifiers, Random Forests are attractive because they (i) reduce variance through bootstrap aggregating (bagging), (ii) inject feature randomness at each split to decorrelate trees, (iii) require little feature scaling, and (iv) yield interpretable feature-importance rankings [3], [4].

**Motivation.** In clinical screening contexts, false negatives (missed malignancies) are especially costly, while false positives create unnecessary follow-up burden. Our goal is to develop a strong, transparent baseline Random Forest for WDBC that generalizes to a locked holdout set and surfaces which nucleus measurements drive predictions.

**Contributions.** This work (1) establishes a clean train/holdout protocol with stratified sampling; (2) tunes a Random Forest using five-fold stratified cross-validation and GridSearchCV; (3) reports a full metric suite (accuracy, precision, recall, F1, MCC, ROC-AUC) with visualizations; and (4) analyzes feature importance and error modes to guide future recall-oriented improvements.

---

## II. Related Work / Literature Review

Ensemble tree methods are widely used in biomedical tabular prediction. Breiman’s Random Forest algorithm combines bagging with random feature selection to improve generalization over single trees [3]. Subsequent surveys summarize ensemble practice and evaluation protocols in clinical ML [4], [5].

On breast cancer tabular datasets, prior studies report strong performance for Random Forests, Support Vector Machines (SVMs), and boosting methods on WDBC and related corpora [6], [7]. Comparative pipelines typically include stratified cross-validation, hyperparameter search, and reporting of sensitivity/specificity alongside accuracy because class imbalance and asymmetric clinical costs matter [8].

Our methodology is also informed by published Random Forest clinical workflows that emphasize five-fold CV, GridSearchCV, and multi-metric evaluation (accuracy, precision, recall, F1, MCC, ROC-AUC), including recent risk-stratification studies outside oncology that share the same modeling template [9]. Explainability tools such as SHAP have become common for interpreting ensemble decisions in medicine [10]. Deep learning dominates imaging-based breast cancer CAD [11], but for small-to-medium tabular FNA feature sets, classical ensembles remain competitive and more sample-efficient [12].

Relative to this literature, our paper focuses on a carefully locked holdout evaluation, unscaled features for trees, and a detailed discussion of false-negative residual errors on WDBC.

---

## III. Methodology

### A. Problem Formulation

Let \(\mathbf{x} \in \mathbb{R}^{30}\) denote a tumor’s feature vector and \(y \in \{0,1\}\) the diagnosis label (0 = benign, 1 = malignant). We learn a classifier \(f:\mathbb{R}^{30}\rightarrow\{0,1\}\) minimizing classification error under stratified validation, then evaluate once on holdout data.

### B. Random Forest Classifier

A Random Forest is an ensemble of \(B\) decision trees \(\{T_b\}_{b=1}^{B}\). For each tree \(b\):

1. **Bootstrap sampling (bagging):** Draw a training set \(D_b\) of size \(n\) from the development data **with replacement**.
2. **Tree growth with feature randomness:** At each split, randomly select \(m=\sqrt{p}\) features (\(p=30\)) and choose the best split among them (Gini impurity by default in scikit-learn).
3. **Prediction:** For a new \(\mathbf{x}\), each tree votes a class; the forest returns the majority vote
   \[
   \hat{y} = \mathrm{mode}\big(T_1(\mathbf{x}),\ldots,T_B(\mathbf{x})\big).
   \]
   Class probabilities are estimated by the fraction of trees voting for each class.

**Class weighting.** We set `class_weight="balanced_subsample"` so class weights are computed from each tree’s bootstrap sample, mitigating mild imbalance between benign and malignant cases.

### C. Hyperparameter Tuning and Cross-Validation

We use **five-fold stratified cross-validation** on the development set. In fold \(k\), the model trains on four folds and validates on the held-out fold. GridSearchCV searches:

- `n_estimators` \(\in \{100,200,300\}\)
- `max_depth` \(\in \{\mathrm{None},5,10,20\}\)
- `min_samples_split` \(\in \{2,5\}\)
- `min_samples_leaf` \(\in \{1,2\}\)
- `max_features` \(\in \{\texttt{sqrt},\texttt{log2}\}\)

Selection metric: **accuracy**. The winning configuration is refit on the full development set.

**Best model:** \(B=300\), `max_depth=None`, `max_features=sqrt`, `min_samples_split=5`, `min_samples_leaf=1`, `random_state=42`.

### D. Relationship Between Bootstrap and Cross-Validation

Cross-validation is the **outer** evaluation/tuning loop. Bootstrapping occurs **inside** each Random Forest fit: whenever trees are trained (including within each CV fold), each tree receives its own bootstrap sample. After CV, mean and standard deviation of fold metrics summarize statistical performance; bagging itself aggregates tree votes rather than computing a bootstrap mean of a scalar statistic.

### E. Workflow Summary

WDBC clean data \(\rightarrow\) stratified 80/20 split \(\rightarrow\) GridSearchCV (5-fold) on development \(\rightarrow\) best RF model \(\rightarrow\) holdout evaluation + feature importance.

*(See also project workflow and model diagrams in the repository figures.)*

---

## IV. Dataset Description

### A. Source and Features

We use the Wisconsin Diagnostic Breast Cancer dataset [1], [2]. After cleaning (`wdbc_clean.csv`), the data contain:

- **569** samples (patients/tumors)
- **30** continuous features: mean, standard error, and “worst” (largest mean of the three largest values) for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension
- **Label:** diagnosis originally \(\{M,B\}\), encoded as \(\{1,0\}\)

No missing values were present; patient IDs are unique.

### B. Preprocessing and Splits

1. Encode diagnosis to integers; drop ID from the feature matrix.
2. **Stratified 80/20 split** preserving class proportions:
   - Development: **455** samples → `development_unscaled.csv`
   - Final holdout: **114** samples → `test_unscaled_FINAL_HOLDOUT.csv` (**locked**; unused during tuning)
3. Features remain **unscaled** (tree splits are scale-invariant).

### C. Exploratory Data Analysis (EDA)

EDA figures are provided in `report_figures/` (also copied into the submission package):

![Class distribution](report_figures/eda_class_distribution.png)

![Stratified split counts](report_figures/eda_split_counts.png)

![Feature boxplots by diagnosis](report_figures/eda_feature_boxplots.png)

![Correlation heatmap](report_figures/eda_correlation_heatmap.png)

Qualitatively, malignant cases show larger area/perimeter and higher concave-points values, consistent with irregular, enlarged nuclei.

---

## V. Experiments, Results, and Discussion

### A. Experimental Setup

- **Library:** scikit-learn `RandomForestClassifier`, `GridSearchCV`, `StratifiedKFold`
- **Hardware/software:** Python 3, pandas, NumPy, joblib, matplotlib
- **Primary scripts/notebooks:** `train_random_forest_model.py`, `Random_Trees.ipynb`, `03_Model_Preprocessing.ipynb`
- **Note on training loss:** Random Forests are not trained by iterative gradient descent; there is **no epoch-wise training loss curve**. We instead report **per-fold CV accuracies** and holdout metrics as learning/evaluation diagnostics.

### B. Metrics Descriptions

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Fraction of correct predictions overall |
| **Precision (malignant)** | \(TP/(TP+FP)\): of predicted malignancies, how many were truly malignant |
| **Recall (malignant)** | \(TP/(TP+FN)\): of true malignancies, how many were detected |
| **F1** | Harmonic mean of precision and recall |
| **MCC** | Correlation of predictions vs labels using TP,TN,FP,FN; range \([-1,1]\) |
| **ROC-AUC** | Ranking quality of predicted probabilities across thresholds |

### C. Cross-Validation Results

**TABLE I. Five-fold stratified CV metrics on development data (n=455)**

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.9648 | 0.0128 |
| Precision | 0.9536 | 0.0219 |
| Recall | 0.9529 | 0.0300 |
| F1 | 0.9528 | 0.0172 |
| ROC-AUC | 0.9885 | 0.0073 |

Fold accuracies were stable (approximately 0.945–0.978), indicating low selection variance.

### D. Holdout Results

**TABLE II. Locked holdout performance (n=114)**

| Metric | Value |
|--------|-------|
| Accuracy | **0.9737** |
| Precision (malignant) | **1.0000** |
| Recall (malignant) | **0.9286** |
| F1 | **0.9630** |
| MCC | **0.9442** |
| ROC-AUC | **0.9964** |

**TABLE III. Holdout confusion matrix**

|  | Pred. Benign | Pred. Malignant |
|--|-------------:|----------------:|
| **True Benign** | 72 | 0 |
| **True Malignant** | 3 | 39 |

### E. Visualization of Results

Figures in `report_figures/` include:

![CV accuracy by fold](report_figures/results_cv_fold_accuracy.png)

![CV metrics mean ± std](report_figures/results_cv_metrics.png)

![Holdout confusion matrix](report_figures/results_confusion_matrix.png)

![Holdout ROC curve](report_figures/results_roc_curve.png)

![Holdout metrics](report_figures/results_holdout_metrics.png)

![Feature importance](report_figures/results_feature_importance.png)

### F. Feature Importance

Top predictors: `area_worst` (0.146), `perimeter_worst` (0.142), `concave_points_worst` (0.103), `concave_points_mean` (0.100), `radius_worst` (0.079). Worst-case size and concavity features dominate, aligning with clinical intuition that extreme irregular morphology indicates malignancy.

### G. Results Discussion

The model generalizes well: holdout accuracy (97.37%) slightly exceeds CV mean (96.48%), suggesting the locked split is not adversely biased. Perfect malignant precision (no false positives) is desirable for avoiding unnecessary alarm, but recall of 0.93 implies **three missed malignancies**—the primary clinical weakness. Because Random Forest optimizes impurity-based splits rather than an explicit recall loss, future work should incorporate cost-sensitive thresholds or class-specific penalties.

**Baseline comparison context.** At midpoint, logistic regression, SVM, and boosting were planned as formal baselines on the same holdout [project plan]. In this report, Random Forest is positioned against a single decision tree conceptually (bagging + feature randomness reduce variance) and against linear models (RF captures nonlinear interactions among correlated morphological features without manual feature crosses). Completing numerical baseline tables remains recommended for an extended journal version.

---

## VI. Conclusion and Future Directions

We presented a tuned Random Forest for breast cancer diagnosis on WDBC with rigorous stratified CV and locked holdout evaluation. The final model attains **97.37% holdout accuracy** and **0.996 ROC-AUC**, with transparent feature importances centered on worst-area and concave-points measurements.

**Future directions:** (1) optimize decision thresholds / class weights to reduce false negatives; (2) run matched baselines (logistic regression, SVM, XGBoost) on the same split; (3) probability calibration; (4) SHAP explanations for case-level interpretability; (5) external validation beyond WDBC.

---

## VII. Code Availability

Code, notebooks, cleaned CSVs, saved model, metrics JSON, and figures are available in the project repository:

**GitHub:** `https://github.com/<YOUR_USERNAME>/<YOUR_REPO>`  
*(Replace with your actual repository URL before submission.)*

**How to run:** see `HOW_TO_RUN.md` and the repository `README.md` in the same Dropbox submission.

Primary entry points:

- `03_Model_Preprocessing.ipynb` — cleaning/split overview  
- `train_random_forest_model.py` or `Random_Trees.ipynb` — training + evaluation  
- Artifacts: `models/random_forest_cv5_model.joblib`, `random_forest_cv5_results.json`

---

## References

[1] W. N. Street, W. H. Wolberg, and O. L. Mangasarian, “Nuclear feature extraction for breast tumor diagnosis,” in *Proc. SPIE*, vol. 1905, 1993, pp. 861–870.

[2] W. H. Wolberg, W. N. Street, and O. L. Mangasarian, “Machine learning techniques to diagnose breast cancer from fine-needle aspirates,” *Cancer Lett.*, vol. 77, no. 2–3, pp. 163–171, 1994.

[3] L. Breiman, “Random forests,” *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[4] T. G. Dietterich, “Ensemble methods in machine learning,” in *Multiple Classifier Systems*. Berlin, Germany: Springer, 2000, pp. 1–15.

[5] M. Fernández-Delgado, E. Cernadas, S. Barro, and D. Amorim, “Do we need hundreds of classifiers to solve real world classification problems?” *J. Mach. Learn. Res.*, vol. 15, pp. 3133–3181, 2014.

[6] A. F. M. Agarap, “On breast cancer detection: An application of machine learning algorithms on the Wisconsin Diagnostic Dataset,” *arXiv:1711.07831*, 2017.

[7] K. Kourou, T. P. Exarchos, K. P. Exarchos, M. V. Karamouzis, and D. I. Fotiadis, “Machine learning applications in cancer prognosis and prediction,” *Comput. Struct. Biotechnol. J.*, vol. 13, pp. 8–17, 2015.

[8] J. A. Hanley and B. J. McNeil, “The meaning and use of the area under a receiver operating characteristic (ROC) curve,” *Radiology*, vol. 143, no. 1, pp. 29–36, 1982.

[9] (Methodological reference used in midpoint planning) Random Forest clinical prediction workflow with 5-fold CV and GridSearchCV as adapted from related biomedical ensemble studies; see PubMed record 40615474 for an example RF risk-stratification pipeline template.

[10] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Proc. NeurIPS*, 2017.

[11] L. Shen *et al.*, “Deep learning to improve breast cancer detection on screening mammography,” *Sci. Rep.*, vol. 9, Art. no. 12495, 2019.

[12] R. Shwartz-Ziv and A. Armon, “Tabular data: Deep learning is not all you need,” *Inf. Fusion*, vol. 81, pp. 84–90, 2022.

[13] F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[14] D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” *BMC Genomics*, vol. 21, Art. no. 6, 2020.
