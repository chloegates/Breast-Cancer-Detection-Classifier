# Breast Cancer Random Forest — Diagrams

## 1. What the model is doing

Each new tumor enters the forest as a **30-feature vector** (unscaled).  
300 decision trees each make an independent prediction. The final class is chosen by **majority vote**.

```mermaid
flowchart LR
    X["Input x<br/>30 tumor features<br/>(unscaled)"]

    subgraph RF["Random Forest (300 trees)"]
        T1["Tree 1<br/>if area_worst <= ..."]
        T2["Tree 2<br/>if perimeter_worst <= ..."]
        T3["Tree 3<br/>if concave_points <= ..."]
        TN["Tree 300<br/>..."]
    end

    V["Votes<br/>Benign vs Malignant"]
    Y["Final prediction<br/>+ confidence %"]

    X --> T1
    X --> T2
    X --> T3
    X --> TN
    T1 --> V
    T2 --> V
    T3 --> V
    TN --> V
    V --> Y
```

### Inside one tree
- Trained on a **bootstrap sample** of the development data
- At each split, only a **random subset of features** is considered
- Leaf node outputs **Benign (0)** or **Malignant (1)**

### Your trained model
- Best params: `n_estimators=300`, `max_depth=None`, `max_features='sqrt'`
- 5-fold CV accuracy: **96.48%**
- Holdout accuracy: **97.37%**

---

## 2. Project workflow

```mermaid
flowchart LR
    A["wdbc_clean.csv<br/>569 samples"] --> B["Clean & encode<br/>30 features + diagnosis"]
    B --> C["80/20 split"]
    C --> D["development_unscaled.csv<br/>455 samples"]
    C --> E["test_unscaled_FINAL_HOLDOUT.csv<br/>114 samples<br/>(locked)"]
    D --> F["5-fold CV + GridSearchCV"]
    F --> G["Best Random Forest model<br/>random_forest_cv5_model.joblib"]
    G --> H["Evaluate on holdout"]
    E --> H
    H --> I["Metrics + feature importance<br/>+ midpoint report"]
```

## Image files
- `breast_cancer_random_forest_model_diagram.png`
- `breast_cancer_project_workflow.png`
