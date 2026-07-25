# People Analytics at Teach For America — Recruitment Funnel Prediction

**MAX 522: Predictive Analytics**
Adam Bashir, Adina Asanova, Barbie Jindal, Indrani Grewal, Rachel Lummis, Victoria Vann

---

## Business Problem

Teach For America recruits and trains college graduates to teach in under-resourced schools. Applications peaked at ~57,000 in 2013, then declined 12% (2014), 10% (2015), and 16% (2016) — while limited recruiter staff kept spending time on applicants who never completed the process.

The 6-stage admissions funnel (online application → document submission → interview → selection/offer → region placement → training) had five identified friction points:

- Excessive upfront effort in a long application → high abandonment
- High investment, low applicant confidence early on
- Transcripts/references requested too early
- Recruiter contact came too late in the process
- A protracted 4–6 week timeline

## Our Goals

1. Build a model to identify high-potential applicants, so recruiter resources can be allocated efficiently
2. Streamline the recruitment process itself to improve the applicant experience

---

## Approach

### Feature Engineering
- Removed 20+ leakage, high-cardinality, and redundant fields (post-decision variables, operational timestamps, duplicate cleaned/uncleaned columns, university names)
- Engineered an **early-submission** feature (days between submission and deadline)
- Engineered a **sentiment-score feature** (`Essays.Sentiment`) from applicant essay text
- Ran **k-means clustering (k=3)** on cumulative GPA to build Low/Medium/High academic tiers, used for missing-value imputation
- Final dataset: 18+ predictors spanning academic, behavioral, essay-content, and categorical variables, with an 80/20 train-test split

### Models Compared
Five classification models were built and evaluated on accuracy, Kappa, sensitivity, and specificity against a class-imbalanced target (~80% complete / ~20% incomplete):

| Model | Notes |
|---|---|
| **KNN** | Selected as the best-performing model — highest accuracy, highest Kappa, and the best balance of sensitivity/specificity |
| ANN | Neural network (`nnet`), hyperparameter-tuned via cross-validation (15-node hidden layer) |
| Naive Bayes | Built for comparison on the same cleaned dataset; high specificity but low Kappa/sensitivity, did not outperform KNN |
| SVM | Built for comparison on the same cleaned dataset; high specificity but low Kappa/sensitivity, did not outperform KNN |
| Decision Tree | Built for comparison on the same cleaned dataset; high specificity but low Kappa/sensitivity, did not outperform KNN |

The R script in this repo covers the data cleaning, feature engineering, and the KNN + ANN modeling pipeline in detail. The full 5-model comparison and final selection are documented in the accompanying team presentation (`TFA_Final_Presentation.pdf`).

---

## Recommendation: Redesigned Recruitment Funnel

Based on the model and funnel diagnosis, we proposed restructuring the process to intervene earlier:

**Old funnel:** Online Application → Document Submission → Interview & Demo Lesson → Selection & Region Placement → Offer Acceptance & Training

**New funnel:** Online Application → **Follow-up Call/Email for incomplete applications** → Interview & Demo Lesson → Conditional Offer (references/transcripts requested here, later in the process) → Offer Acceptance, Training & Placement

Moving document requests later and adding an early follow-up touchpoint targets the two biggest friction points: costly early asks and late recruiter contact.

## Business Impact

- **Higher-quality data** with less friction in early application stages
- **Better resource allocation** — recruiters can prioritize applicants by predicted completion likelihood
- **More completed applications** — the redesigned funnel is built to attract and retain higher-quality candidates

---

## Tech Stack

R (`caret`, `nnet`, `NeuralNetTools`, `fastDummies`), k-means clustering, KNN, ANN

## Project Structure

```
People-Analytics-TFA/
├── People_Analytics_TFA_Admissions_Prediction.R   # Data cleaning, feature engineering, KNN + ANN pipeline
├── TFA_Final_Presentation.pdf                     # Final team deck: problem, approach, 5-model comparison, funnel redesign, business impact
└── README.md
```

## How to Run

```r
install.packages(c("caret", "tidyverse", "nnet", "fastDummies", "NeuralNetTools", "C50", "e1071"))
# Update the file path in the script to your local copy of the dataset, then run top to bottom.
```

