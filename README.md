# Customer Churn Prediction: From Data Decisions to Targeted Outreach

> A churn score is only useful if you know how to use it. This project investigates the data choices behind churn prediction, compares three model families, and asks a practical question: if a retention team can contact only a fraction of its customers, how well can each model prioritize that list?

---

## Key Findings

- **Early tenure is the highest-risk window.** Risk is highest in the first month, falls fastest over the first six months, and tenure only becomes a protective factor after roughly 17 months.
- **Contracts are the strongest protective signal.** Holding other factors constant, two-year customers have about **1/4 the odds** of churning compared with month-to-month customers; one-year customers have about half.
- **Contacting the riskiest 20% of customers finds about half of all churners**, at 68% precision, or roughly **2.6× better than random targeting**.
- **Gradient boosting's edge over logistic regression is small and concentrated at the top of the list.** It identified 8 more churners in the top 5%, but the two models were identical at 20% and 30% capacity.
- **Removing redundant billing features cost nothing for logistic regression and boosting, but hurt random forest.** The same data decision can have different consequences for different model families.

---

## What This Project Does

| Step | What I did |
|---|---|
| 🧹 **Investigated missing and redundant data** | Traced missing billing values to their cause and verified overlapping service categories row by row |
| 🧪 **Tested feature choices across models** | Compared four feature sets across three model families to see whether redundant billing information still helps prediction |
| 📊 **Compared model families on shared splits** | Evaluated logistic regression, random forest, and gradient boosting on identical cross-validation folds |
| 🎯 **Evaluated limited outreach capacity** | Measured churners identified in the top 5%, 10%, 20%, and 30% of each risk-ranked list |
| 🔍 **Explained predictions and trade-offs** | Compared logistic regression coefficients with SHAP, and examined an illustrative cost scenario |

---

## The Problem

A retention team has limited time and resources. It needs to decide which customers to contact, and whether a more complex model makes that decision meaningfully better.

That takes more than comparing overall model scores. A model might perform slightly better on average without identifying more churners at the contact budget the team actually has. Before trusting either model, there are also data decisions to make: how to handle missing values, which categories repeat the same information, and which features are worth keeping.

This project connects those choices to the final customer shortlist:

1. **Understand the data:** investigate missing billing records and service relationships.
2. **Test the assumptions:** compare feature sets across different model families.
3. **Evaluate the decision:** measure performance at fixed outreach capacities.

The shortlist identifies customers with higher observed churn risk. Whether an offer or a support call would persuade them to stay is a separate question that requires an intervention experiment (see [Limitations](#limitations-and-next-steps)).

---

## Project Evolution

The [first version of this project](LINK-TO-R-VERSION), built in R, compared six classifiers and found that logistic regression performed nearly as well as boosting. It also explored churn patterns, SHAP explanations, and thresholds under assumed misclassification costs.

That gave me a useful baseline, and a few questions I wanted to revisit:

- **Is the data really as clean as it looks?** I had dropped missing records without asking why they were missing.
- **Which features carry independent information?** I had kept overlapping billing features without testing whether each model needed them.
- **What does a model score mean for the team using it?** I wanted to move beyond overall metrics and evaluate the customer list a team with a fixed contact budget would actually receive.

This Python version is built around those questions.

| | R version (v1) | Python version (v2) |
|---|---|---|
| **Missing values** | Dropped | Traced to zero-tenure customers; filled as "not yet billed" |
| **Repeated service categories** | Kept as-is | Verified row by row, then merged |
| **Billing features** | Kept | Tested four feature sets across three model families |
| **Models** | Six classifiers | Three model families, with focused tuning of boosting |
| **Evaluation** | ROC-AUC, confusion matrix | ROC-AUC, average precision, fixed-capacity ranking metrics |
| **Business framing** | Cost-based threshold | Cost-based threshold and outreach capacity |

Headline performance barely changed between versions, and that was itself a finding: the first version's conclusions held up. What changed is that every step is now validated, and the reasoning became more specific. **Redundant features affected model families differently, and the advantage of one model depended on how many customers the team could contact.**

The original R analysis is retained as the initial version. Everything below refers to the Python version.

---

## The Data

**7,043 customers · 19 original predictors · 26.5% churn rate**

The project uses the IBM Telco Customer Churn sample dataset. Each row represents a customer, with information about:

- **Customer characteristics:** gender, senior citizen status, partners, and dependents
- **Subscribed services:** phone, internet, security, support, and streaming
- **Billing and contracts:** payment method, contract type, monthly charges, and total charges
- **Customer tenure:** months with the company

The original class distribution is preserved. Instead of resampling, class imbalance is handled at the decision stage through thresholds and ranking, so predicted probabilities stay on their natural scale. Stratified splitting (70/30) and stratified 10-fold cross-validation keep the churn rate consistent across evaluation subsets. The test set was used once, for final evaluation.

This is a single customer snapshot. It supports an offline modeling exercise, but does not establish how the models would perform on future customer cohorts.

---
## Data Decisions: Understanding the Customer Behind the Row

Before comparing models, I wanted to know what the data was actually recording. A missing bill, a repeated service label, and a suspiciously well-correlated feature each told a different story, and each called for a different call.

### 1. Missing total charges: a billing-cycle question, not a data-quality one

`TotalCharges` loaded as text. Converting it to numeric surfaced 11 whitespace-only entries that `isnull()` had quietly missed.

The pattern was too clean to be random. All 11 belonged to customers with zero tenure, and every zero-tenure customer had a blank. These customers already had a plan price (`MonthlyCharges`) but hadn't hit their first billing cycle yet. So "missing" here really meant "not billed yet," and I filled these values with 0.

Dropping them would have been the easy default (and it's what I did in the first version). But it's the wrong instinct for a churn problem. New customers turned out to be the highest-risk group in the entire dataset, so a pipeline that quietly discards them is removing exactly the people a retention team most needs to see. With 11 rows it barely moves the metrics. In a live system where new sign-ups arrive daily, it would be a real blind spot.

### 2. Repeated service labels: keep the information, drop the duplication

Six add-on fields contained `"No internet service"`, and `MultipleLines` contained `"No phone service"`. Matching counts weren't enough for me, so I checked row by row: these labels flagged exactly the same customers as `InternetService = No` and `PhoneService = No`.

I merged them into `"No"` and kept the parent service fields. The model can still tell "has internet but skipped tech support" apart from "has no internet at all," without seeing the same fact seven times. That duplication would have created perfect collinearity for logistic regression and scattered SHAP credit across redundant columns.

The distinction matters beyond the math. "Didn't buy" and "couldn't buy" are different business situations: the first is a potential upsell, the second is a customer who was never offered the product.

### 3. Billing features: does redundant information still earn its place?

Two checks changed how I saw the billing fields:

- Service indicators explained **99.88% of the variation in monthly charges**.
- Total charges correlated **0.9996** with `tenure × MonthlyCharges`.

Monthly charges, in other words, is basically the price tag of the service bundle, and total charges is that price accumulated over time. Pairwise correlations understated this: total charges looked only moderately related to each input alone (0.83 with tenure, 0.65 with monthly charges) but was almost exactly their product.

This also answered a business question I'd wanted to ask. Fiber customers churn more, but they also pay more. Is it the price or the product? In this dataset the two are nearly inseparable, so it can't be answered here. Separating them would take network-quality data or a pricing test.

Still, recoverable information isn't necessarily useless information. A model might benefit from having a summary handed to it directly, even if it could in theory rebuild it. Redundancy is a property of the data; whether removing it costs anything depends on the model. So rather than applying a correlation cutoff, I tested it: four feature sets, three model families, identical cross-validation folds.

- **Logistic regression and gradient boosting:** no meaningful difference. The gaps were about a tenth of the fold-to-fold noise. I dropped both billing fields for a simpler, fully interpretable model.
- **Random forest:** it did rely on them, losing 0.013 ROC-AUC and 0.034 average precision without them. It also trailed the other two models in every version, so it didn't change the final choice.

The takeaway: **the same data decision can be harmless for one model and costly for another.** That's why I tested it per model family instead of deciding once from the correlation matrix.

Dropping monthly charges as a *feature* doesn't mean discarding it. It's still the right measure of what a customer is worth when prioritizing outreach.
> **Do billing features add enough predictive value to justify keeping them—and does the answer depend on the model?**

---

## Modeling: Testing the Feature Decision

I compared four feature sets across logistic regression, random forest, and gradient boosting, using fixed baseline configurations and the same stratified 10-fold cross-validation splits.

Logistic regression used L2 regularization, with standardization fitted within each training fold.

| Feature set | LR ROC-AUC | LR AP | RF ROC-AUC | RF AP | GBM ROC-AUC | GBM AP |
|---|---|---|---|---|---|---|
| A: all features | 0.845 | 0.655 | 0.827 | 0.630 | 0.846 | 0.664 |
| B: drop `TotalCharges` | 0.843 | 0.653 | 0.822 | 0.610 | 0.848 | 0.662 |
| **C: drop both billing fields** | **0.843** | **0.653** | 0.814 | 0.596 | **0.847** | **0.666** |
| D: drop `MonthlyCharges` | 0.845 | 0.655 | 0.825 | 0.623 | 0.847 | 0.666 |

*Mean cross-validation scores. AP = average precision.*

### What changed—and what did not

**LR and GBM retained nearly the same ranking performance without the billing fields.** For LR, removing both reduced mean ROC-AUC by just 0.0019 and AP by 0.0018. GBM achieved its highest mean AP with both removed.

I carried forward **feature set C**, accepting the small observed trade-off for LR in exchange for a leaner input set with less billing-related redundancy.

**Random forest told a different story.** Removing both fields reduced ROC-AUC by 0.0133 and AP by 0.0340. Its mean scores favored keeping all features.

One plausible explanation is that billing fields give trees convenient continuous summaries of the customer’s service bundle and tenure. Reconstructing those summaries through separate splits may be harder under the baseline configuration.

RF also trailed LR and GBM in this initial comparison, so I focused the next stage on those two models.

### The decision

The useful finding was not simply that two columns could be removed. It was that **the same redundancy had different consequences across model families**.

EDA identified the overlap. The ablation experiment showed whether removing it mattered. That gave me a stronger basis for feature selection than a correlation threshold alone..

### Tuning and final comparison

Gradient boosting was tuned with a grid search over tree count, depth, and learning rate on feature set C. The best model used **depth-1 trees** (300 trees, learning rate 0.1).

Depth-1 trees split on a single variable each, so the model is **additive, with no interactions**. And because every feature in set C except tenure is binary, where a linear effect and a nonlinear one are the same thing, boosting's edge over logistic regression can only come from one place: **the nonlinear shape of tenure**. The SHAP analysis below confirms this.

| Model | CV ROC-AUC | Test ROC-AUC | CV AP | Test AP |
|---|---|---|---|---|
| Logistic regression | 0.843 | 0.844 | 0.653 | 0.653 |
| Gradient boosting (tuned) | 0.850 | 0.848 | 0.673 | 0.667 |

Test results closely match cross-validation, so the model selection above generalizes. Average precision of about 0.67 is roughly **2.5× the random baseline** of 0.265.

---

## What Drives Churn

Logistic regression coefficients and gradient boosting SHAP values were computed independently. **Every one of the top nine SHAP features has the same direction in logistic regression, and all are statistically significant (p < 0.01).**

| Factor | Odds ratio (LR) | Direction |
|---|---|---|
| Two-year contract (vs. month-to-month) | 0.25 | Protective |
| No internet service (vs. DSL) | 0.44 | Protective |
| One-year contract (vs. month-to-month) | 0.52 | Protective |
| Online security | 0.67 | Protective |
| Tech support | 0.70 | Protective |
| Each additional month of tenure | 0.97 | Protective |
| Electronic check (vs. bank transfer) | 1.39 | Risk |
| Fiber optic (vs. DSL) | 2.52 | Risk |

![SHAP summary](figures/shap_beeswarm.png)

A few details worth noting:

- **Tenure's effect is strongly nonlinear.** Logistic regression assumes a constant 3% drop in odds per month. SHAP shows risk is highest in the first month, falls steeply through month six, stays elevated until around month 17, and then becomes protective. This is the pattern logistic regression cannot fit, and it explains boosting's edge.

![Tenure SHAP dependence](figures/shap_tenure.png)

- **Not all add-on services behave the same.** Online security and tech support are significant protective factors; online backup and device protection are not.
- **Only electronic check stands out among payment methods.** Mailed check, also a manual method, is not significantly different from automatic bank transfer, so the pattern isn't simply manual vs. automatic payment.
- **Streaming services carry some pricing information.** With monthly charges removed, part of the price signal shifts onto the service indicators. Streaming's positive association with churn should be read with that in mind.
- **Demographics matter little.** Gender, partner status, and dependents were not significant, which points retention efforts toward contracts, services, and early tenure.

These are associations, not causal effects. For example, part of the two-year contract effect likely reflects that customers who already plan to stay are the ones who sign long contracts.

---

## Evaluating the Customer List

### Fixed outreach capacity

If the team can contact only the top k% of customers by predicted risk:

| Capacity | Contacted | LR churners found | GBM churners found | GBM precision | Share of all churners found | Lift vs. random |
|---|---|---|---|---|---|---|
| Top 5% | 106 | 85 | **93** | 0.88 | 17% | 3.3× |
| Top 10% | 212 | 158 | **162** | 0.76 | 29% | 2.9× |
| Top 20% | 423 | 288 | 287 | 0.68 | 51% | 2.6× |
| Top 30% | 634 | 374 | 374 | 0.59 | 67% | 2.2× |

- **The riskiest customers are identified most precisely.** Lift falls from 3.3× at 5% to 2.2× at 30%.
- **Boosting only helps at the very top.** Its advantage is 8 churners at 5% capacity and disappears by 20%. This fits the tenure finding: logistic regression's straight-line fit underestimates the risk of the newest customers, who dominate the top of the list.
- **Model choice depends on capacity.** For a small, focused campaign, boosting has a modest practical edge. For broader outreach, the two models are interchangeable, and logistic regression is simpler to explain.

### An illustrative cost scenario

Under the standard cost-sensitive decision rule, a customer is worth contacting when the expected cost of missing a churner exceeds the cost of an unnecessary contact. Assuming a missed churner costs **10×** an unnecessary contact, the threshold is 1/11 ≈ **0.09**, compared with the default of 0.5.

| Model | Threshold | Recall | Precision | Total cost |
|---|---|---|---|---|
| LR | 0.50 | 0.53 | 0.67 | 2,796 |
| LR | 0.09 | 0.94 | 0.39 | 1,163 |
| GBM | 0.50 | 0.50 | 0.70 | 2,932 |
| GBM | 0.09 | 0.95 | 0.39 | 1,131 |

Moving from the default threshold cuts total cost by about 60% and finds nearly every churner. But it means contacting about **64% of all customers**, which few retention teams could do. Compared with the capacity view, going from 30% to 64% outreach more than doubles the contacts to find about a quarter more churners.

This rule assumes reasonably calibrated probabilities, and the result is sensitive to the assumed cost ratio. It is included to show the trade-off, not as a recommended policy.

---

## Limitations and Next Steps

- **Risk is not persuadability.** The model predicts who is likely to leave, not who would stay if contacted. Some high-risk customers may leave regardless, and some low-risk customers might react badly to outreach. The next step is a randomized retention experiment within risk tiers, measuring incremental retention by tier, which could later support an uplift model.
- **Associations, not causes.** Contract, payment, and service effects are observational and may reflect who selects into each option.
- **No time-based validation.** This is a single snapshot, so the model could not be validated on future customers. A production version should train on past periods and test on later ones.
- **Customer value is not considered.** All churners are weighted equally. Ranking by expected revenue at risk (churn probability × monthly charges × a time horizon) may produce a different priority list.
- **Cost assumptions need business input.** The cost ratio used above is illustrative and would need to be grounded in real offer costs and customer lifetime value.

---

## Repository Structure

```
├── README.md
├── churn_analysis.py        # Full Python workflow
├── data/
│   └── TelcoCustomerChurn.csv
├── figures/
│   ├── shap_beeswarm.png
│   └── shap_tenure.png
└── r-version/               # Initial R analysis (v1)
```

## How to Run

```bash
pip install pandas numpy scikit-learn statsmodels shap matplotlib
python churn_analysis.py
```
