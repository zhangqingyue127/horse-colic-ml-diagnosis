# Horse Colic Diagnosis with Machine Learning

A portfolio-ready reconstruction of my **Machine Learning course final project** from the **spring semester of sophomore year** .

This project studies a classic veterinary diagnosis task: predicting horse colic outcomes from clinical indicators using supervised machine learning. The original coursework compared **Logistic Regression**, **Random Forest**, and **AdaBoost**, then performed model tuning and result analysis.

## Why this project matters

Horse colic is a high-risk acute condition in equine medicine. Early diagnosis affects treatment decisions and survival outcomes. This project explores how machine learning can support data-driven clinical screening by learning from structured medical observations.

## Project highlights

- Built an end-to-end **binary classification** pipeline for horse colic diagnosis
- Performed **data preprocessing**, **feature encoding**, **standardization**, and **PCA-based feature analysis**
- Compared three classical ML models:
  - Logistic Regression
  - Random Forest
  - AdaBoost
- Evaluated models with:
  - Confusion matrix
  - Accuracy / Precision / Recall / F1
  - ROC-AUC
- Included hyperparameter tuning for all three model families

## Core dataset setup

According to the original course report, the merged dataset contains:

- **366 records**
- **22 columns**
- **21 clinical features + 1 target label**

Representative features include:

- Rectal temperature
- Pulse
- Respiratory rate
- Pain level
- Mucous membrane color
- Capillary refill time
- Packed cell volume
- Total protein
- Abdominocentesis findings

## Main modeling workflow

1. Load training and test text files
2. Clean empty rows and assign readable feature names
3. Separate features and labels
4. Detect low-cardinality columns as candidate categorical variables
5. Standardize features and run PCA
6. Retain components covering **85% cumulative explained variance**
7. Select important features from principal component loadings
8. Encode categorical features
9. Train baseline models
10. Tune model hyperparameters and compare results

## Results summary

### Baseline results from the original notebook

| Model | AUC | Accuracy | Notes |
|---|---:|---:|---|
| AdaBoost | 0.8011 | 0.7910 | Strong positive-class recall |
| Random Forest | 0.8064 | 0.7910 | Best overall baseline balance |
| Logistic Regression | 0.7681 | 0.7313 | Most interpretable model |

### Best tuning results reported in the coursework

| Model | Best CV AUC | Best setting / takeaway |
|---|---:|---|
| Random Forest | **0.8391** | Best overall tuned model |
| AdaBoost | 0.8099 | Strong for high-risk case screening |
| Logistic Regression | 0.7730 | Interpretable but weaker on nonlinear patterns |

## Conclusion

The project conclusion is consistent across the notebook and report:

- **Random Forest** achieved the strongest overall performance and was recommended as the main diagnostic model
- **AdaBoost** showed strong recall for positive/high-risk cases and is useful for preliminary screening
- **Logistic Regression** remained valuable for interpretability and fast deployment

## Repository structure

```text
horse-colic-ml-diagnosis/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ src/
│  └─ horse_colic_ml_pipeline.py
├─ notebooks/
│  ├─ horse_colic_portfolio_version.ipynb
│  └─ horse_colic_course_project_original.ipynb
├─ assets/
│  ├─ pca_explained_variance.png
│  ├─ baseline_confusion_matrices.png
│  ├─ baseline_roc_curves.png
│  ├─ tuned_model_metrics.png
│  └─ tuned_model_comparison.png
├─ data/
│  └─ README.md
├─ docs/
│  ├─ project_summary.md
│  └─ course_report_original_zh.docx
└─ results/
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place the original dataset files in the `data/` directory:

```text
data/
├─ horseColicTraining2.txt
└─ horseColicTest2.txt
```

### 3. Run the cleaned pipeline

```bash
python src/horse_colic_ml_pipeline.py --train data/horseColicTraining2.txt --test data/horseColicTest2.txt --output-dir results
```

## Notes

- This repository is a **GitHub-friendly portfolio version** of the original course submission.
- The original Chinese course report is preserved in `docs/`.
- The original notebook is also retained for archival purposes.
- Some figures in the original coursework were presentation-oriented; this repo keeps the main usable outputs and a cleaner runnable script.

## Course context

This project was completed as the **final assignment for the Machine Learning course** during my **sophomore spring semester** in the **Artificial Intelligence major**.

## Disclaimer

This repository is intended for **academic portfolio presentation and learning purposes**.
