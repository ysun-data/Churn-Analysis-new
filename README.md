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

The [first version of this project](https://github.com/ysun-data/Telecom-Churn-Analysis), built in R, compared six classifiers and found that logistic regression performed nearly as well as boosting. It also explored churn patterns, SHAP explanations, and thresholds under assumed misclassification costs.

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

I looked at churn drivers from two angles that share no machinery: SHAP values from gradient boosting, and odds ratios from logistic regression. They agree. Every one of boosting's top nine features points the same direction in logistic regression, and all are significant at p < 0.01. When a tree model and a linear model tell the same story independently, I trust the story a lot more.

<img width="800" height="550" alt="SHAP summary" src="https://github.com/user-attachments/assets/80b1f974-c154-4249-bea5-3c6cec2bc104" />

*How to read it: each row is a feature, ranked by importance. Red means the feature is high (or "Yes"). Dots right of zero push churn risk up; dots left of zero push it down.*

| SHAP rank | Factor | Odds ratio (LR) | Direction |
|---|---|---|---|
| 1 | Tenure (per additional month) | 0.97 | Protective |
| 2 | Two-year contract (vs. month-to-month) | 0.25 | Protective |
| 3 | Fiber optic (vs. DSL) | 2.52 | Risk |
| 4 | No internet service (vs. DSL) | 0.44 | Protective |
| 5 | One-year contract (vs. month-to-month) | 0.52 | Protective |
| 6 | Streaming movies | 1.44 | Risk |
| 7 | Electronic check (vs. bank transfer) | 1.39 | Risk |
| 8 | Paperless billing | 1.33 | Risk |
| 9 | Online security | 0.67 | Protective |

<details>
<summary><b>Why two versions of logistic regression?</b></summary>

The model used for prediction is L2-regularized on standardized features, which helps it generalize. For this table, I refit the same features without regularization: penalized coefficients are shrunk toward zero and don't come with valid p-values, and unscaled features give effects in units a business can use, like "per month of tenure." Because the regularization was light, both versions produced nearly identical coefficients, so this table describes essentially the same model evaluated above.

</details>

### 1. Tenure: the first six months decide a lot

<img width="600" height="500" alt="Tenure SHAP dependence" src="https://github.com/user-attachments/assets/df5f4616-4305-4398-a9cc-52e5e639347a" />

Logistic regression assumes tenure lowers churn odds by a steady ~3% per month. The SHAP curve says otherwise: risk peaks in the first month, drops steeply through month six, stays above neutral until around month 17, and only then turns protective. That curve is what boosting captures and logistic regression can't, and it's why boosting ranks the newest, riskiest customers more precisely at the top of the list.

For a retention team, this makes timing as important as targeting. An offer to a three-year customer is mostly wasted. The window that matters is the first few months, so I'd schedule the first check-in in month one or two and put any contract-upgrade offer before month six.

### 2. Contracts: the strongest lever the business controls

Two-year customers have about a quarter of the churn odds of month-to-month customers; one-year customers about half. Some of that is selection, since customers who already plan to stay are the ones who sign long contracts. Still, contract type is one of the few drivers the company directly controls. Combined with the tenure finding, it points to a clear test: offer early-tenure, month-to-month customers an incentive to switch to an annual plan, and measure whether retention actually improves.

### 3. Fiber: price or product?

Fiber customers have about 2.5 times the churn odds of DSL customers, the largest risk factor in the model. The obvious question is whether they leave because fiber costs more or because the service disappoints. This dataset can't separate the two, since monthly charges is almost entirely determined by the service bundle (see [Data Decisions](#data-decisions-understanding-the-customer-behind-the-row)). But when the premium product has the least loyal customers, that's worth investigating with network-quality or complaint data.

### Smaller patterns worth a second look

- **Support beats storage.** Online security and tech support are protective; online backup and device protection aren't. My read: services that give customers help when something goes wrong build more attachment than services that just store things. A free tech-support trial for new fiber customers would be a cheap way to test that.
- **The electronic-check puzzle.** Only electronic check stands out among payment methods. Mailed check is also manual but isn't riskier than automatic bank transfer, so this isn't simply "autopay vs. manual." Payment failure rates would be the first thing I'd check.
- **Streaming partly stands in for price.** With monthly charges removed, some of the price signal shifts onto the service indicators, so streaming's link to churn is likely part price, part product.
- **Demographics barely matter.** Gender, partner status, and dependents weren't significant. That's good news: the drivers that do matter are ones the business can act on.
---

## Evaluating the Customer List

### Fixed Outreach Capacity

A small improvement in AUC does not tell an outreach team how many more at-risk customers it can reach. To make the comparison concrete, I ranked test customers by predicted risk and evaluated four illustrative contact budgets.

| Capacity | Customers selected | LR churners found | GBM churners found | GBM precision | GBM share of all churners found | GBM lift vs. random |
|---|---|---|---|---|---|---|
| Top 5% | 106 | 85 | **93** | 88% | 17% | 3.3× |
| Top 10% | 212 | 158 | **162** | 76% | 29% | 2.9× |
| Top 20% | 423 | 288 | 287 | 68% | 51% | 2.6× |
| Top 30% | 634 | 374 | 374 | 59% | 67% | 2.2× |

**The strongest concentration of churners was at the top of the list.** Among the 106 customers selected by GBM at 5% capacity, 93 actually churned—a precision of 88%, compared with the test-set churn rate of about 26.5%.

**Boosting's advantage was concentrated at smaller contact budgets.** It identified eight more churners than LR at 5% capacity and four more at 10%. At 20–30%, the two models covered almost the same number of churners, though not necessarily the same customers.

**The operating constraint changes the model discussion.** For a tightly focused campaign, GBM produced the stronger shortlist in this test sample. For broader outreach, coverage was similar, giving more weight to simplicity and ease of explanation.

These lists measure risk-screening performance. Identifying a likely churner does not establish that contacting them will prevent churn.

### An Illustrative Cost Scenario

A fixed contact budget is one way to set the operating point. Another is to assign different costs to classification errors.

I used a simple scenario:

- A missed churner costs **10 units**.
- A false alarm costs **1 unit**.
- Correct classifications have zero cost.

If predicted probabilities are calibrated, the theoretical threshold is:

$$
t = \frac{1}{10 + 1} \approx 0.091
$$

| Model | Threshold | Recall | Precision | Classification cost units |
|---|---|---|---|---|
| LR | 0.50 | 53% | 67% | 2,796 |
| LR | 0.091 | 94% | 39% | 1,163 |
| GBM | 0.50 | 50% | 70% | 2,932 |
| GBM | 0.091 | 95% | 39% | 1,131 |

*Cost = 10 × false negatives + false positives. The original $200/$20 scenario gives the same threshold; dollar totals would be 20 times the values above.*

Lowering the threshold reduced the assumed classification cost by about **60%** and captured approximately **94–95% of actual churners**. The trade-off was workload: roughly **64% of all customers** were flagged.

For GBM, moving from the top 30% list to this lower threshold increased the selected group from **634 to 1,351 customers**, identifying **156 additional churners**.

The practical question becomes: is that additional coverage worth more than doubling the outreach volume?

This is a classification-cost illustration, not an estimate of retention profit. Probability calibration has not yet been assessed, and the scenario does not model whether outreach works or the full cost of delivering it.

---

## Limitations and Next Steps

- **Risk is not persuadability.** A high-risk customer is not necessarily one who can be retained. The next step would be a randomized retention experiment within risk tiers, comparing an intervention with a control group and measuring incremental retention and net value. With enough experimental data, this could support uplift modeling.

- **Associations are not intervention effects.** Contract, payment, and service patterns may partly reflect the customers who choose those options. Changing the option may not produce the difference observed in the data.

- **Validation is based on a single snapshot.** The random holdout evaluates customers from the same dataset, not a future period. A production version would need timestamped features available before the outcome, a defined prediction window, and validation on later periods.

- **Probability quality needs a separate check.** ROC-AUC, AP, and top-k metrics assess ranking. Calibration should be evaluated before treating scores as absolute churn probabilities or using probability-based cost rules.

- **Customer value is not included.** The current ranking prioritizes churn risk equally across customers. A future version could incorporate expected customer value, intervention costs, and estimated intervention effectiveness. Monthly charges alone are not a measure of lifetime value or profit.

- **Operational assumptions are illustrative.** Contact budgets and the 10:1 error-cost ratio were chosen to explore trade-offs. A deployment decision would require actual team capacity, contact eligibility, costs, and business objectives.

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
