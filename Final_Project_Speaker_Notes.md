# Final Project Presentation — Speaker Notes (~12 minutes)

**Project:** Breast Cancer Detection with Random Forest  
**Slides:** `Final_Project_Presentation_Slides.html` (open in Chrome → present fullscreen; ←/→ to navigate; P to print/PDF)  
**Rubric alignment:** Problem · Methods · Dataset · Results (+ baselines) · Future directions · Q&A

---

## Timing guide (±1 min OK)

| Minutes | Slides | Focus |
|--------:|--------|--------|
| 0:00–0:45 | Title | Introduce yourself + project |
| 0:45–2:15 | Problem | Motivation, clinical stakes |
| 2:15–3:30 | Dataset | WDBC, split, unscaled |
| 3:30–5:15 | Methods + Workflow | RF idea + CV/GridSearch |
| 5:15–8:30 | Results + Analysis | CV, holdout, confusion matrix, baselines |
| 8:30–10:15 | Features + Future | Importance + next steps |
| 10:15–12:00 | Conclusion | Recap + invite questions |
| 12:00–15:00 | Q&A | Use Appendix slide if needed |

---

## What to say (concise script cues)

**Title**  
“Today I’ll present a Random Forest classifier for breast cancer detection using the Wisconsin Diagnostic Breast Cancer dataset.”

**Problem**  
“The task is binary classification: benign vs malignant from cell-nucleus measurements. Accuracy matters because false negatives miss cancer and false positives cause unnecessary follow-up. Our goal was a strong, generalizable, interpretable model.”

**Dataset**  
“569 samples, 30 features. We used an 80/20 split: 455 for development and 114 locked holdout. Features are unscaled because tree models don’t need scaling.”

**Methods**  
“We used Random Forest: many trees vote. Tuned with GridSearch and 5-fold stratified CV. Best model: 300 trees, sqrt features, min_samples_split 5.”

**Results**  
“CV accuracy about 96.5%. Holdout 97.4%, ROC-AUC 0.996. Confusion matrix: 72 correct benign, 39 correct malignant, zero false positives, three false negatives.”

**Baselines / Analysis**  
“Compared conceptually to a single tree (higher variance) and linear models (less flexible for nonlinear morphology). Clinically, the three missed malignancies mean we should emphasize recall next.”

**Future**  
“Cost-sensitive learning, stronger baselines like XGBoost, probability calibration, SHAP explanations, and external validation.”

**Close**  
“Random Forest achieved about 97% holdout accuracy with strong discrimination. Happy to take questions.”

---

## Delivery checklist (Presentation Quality)

- High-contrast slides: black text on white (already set)
- Speak to audience, not the screen
- Point to numbers on results slides; don’t read every bullet
- Pause 1–2 seconds after stating holdout accuracy
- If asked something unknown: “We didn’t test that yet; it’s on our future directions list.”

---

## Rubric self-check

| Criterion | Points | Covered? |
|-----------|--------|----------|
| 12 ± 1 minutes | 2 | Timing table above |
| Slide design (high contrast) + required sections | 3 | All sections included |
| Clarity / delivery / professionalism | 3 | Practice cues + checklist |
| Q&A responses | 2 | Appendix + practice answers |
