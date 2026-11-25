# Customer Churn: Business Understanding and Feasibility Report

**Scope:** telecom customer retention, based on the IBM Telco Customer Churn dataset (7,043 customers).
**Companion files:** analysis notebook, `src/` scripts and the tables in `reports/`. Every figure below can be reproduced from them.

> **Reading note.** Money figures are in *relative units* (one retention offer = 1 unit). The dataset has no currency, no cost data and no campaign history, so the financial section is a decision model built on stated assumptions, not a forecast. Replace the assumptions with company data before any investment decision.

---

## 1. Executive summary

* **Problem.** 26.5% of customers (1,869 of 7,043) have left. Because churners pay more than average (74.4 vs 61.3 monthly charge), they account for **30.5% of monthly recurring revenue**, more than their share of the customer base.
* **Where the risk sits.** Churn is concentrated, not random: month-to-month contracts (42.7%), first-year customers (47.4%), electronic-check payers (45.3%), fiber-optic users (41.9%). The combination month-to-month + fiber + electronic check reaches **60.4%**.
* **What the model adds.** A transparent Logistic Regression ranks customers by churn risk (test ROC-AUC 0.84). Contacting the riskiest 20% reaches about **50% of all churners** (2.5x better than random). Probabilities are well calibrated, so they can be read as risk percentages.
* **Is it worth it?** *Conditionally yes.* Under the base assumptions, targeted outreach cuts the expected cost of churn by about **12%** versus doing nothing, while contacting everyone **loses** money. If a lost customer is worth little relative to an offer, or offers rarely work, the programme does not pay (Section 5.2).
* **Main caveat.** The model is as good as this data allows. Adding engineered features did not improve it (Section 4.3). Further accuracy needs new data (usage, complaints, support contacts), and the true retention effect must be measured with an A/B test.
* **Recommendation.** Run a controlled pilot on the High-risk tier, combined with structural fixes (contract migration, auto-pay, first-year onboarding).

---

## 2. Business problem and objectives

**Problem statement.** The operator loses roughly one customer in four. Replacing a customer usually costs more than keeping one, so unmanaged churn erodes revenue and wastes acquisition spend.

**The decision to support.** *Which customers should the retention team contact, with which offer, and is the cost justified?*

| Stakeholder | Need |
|---|---|
| Retention / CRM team | A ranked contact list with a reason for each customer |
| Marketing | Which segments and offers to prioritise |
| Finance | Expected cost, saving and break-even conditions |
| Product / Network | Structural causes (fiber service, tech support, contract design) |

**Objectives and success measures**

| Objective | Measure |
|---|---|
| Identify who is likely to churn | ROC-AUC, PR-AUC, recall at the contact threshold |
| Spend retention budget efficiently | Precision at threshold, expected cost versus alternatives |
| Explain risk to business users | Reason codes per customer, SHAP |
| Prove the programme works | Churn difference between treated and holdout groups (A/B test) |

---

## 3. Business understanding: who churns and why

**Scale.** 1,869 churners out of 7,043 customers. Median tenure of churners is 10 months versus 38 for retained customers.

| Segment | Churn rate | Customers |
|---|---|---|
| Month-to-month contract | 42.7% | 3,875 |
| One-year contract | 11.3% | 1,473 |
| Two-year contract | 2.8% | 1,695 |
| Electronic check payment | 45.3% | 2,365 |
| Automatic payment (bank / card) | 16.0% | 3,066 |
| First 12 months of tenure | 47.4% | 2,186 |
| Tenure 49-72 months | 9.5% | 2,239 |
| Fiber optic internet | 41.9% | 3,096 |
| DSL internet | 19.0% | 2,421 |
| No internet service | 7.4% | 1,526 |
| Internet customers without tech support | 41.6% | 3,473 |
| Senior citizens | 41.7% | 1,142 |
| Month-to-month + fiber + electronic check | 60.4% | 1,307 |

**Interpretation**

1. **Commitment protects.** Contract length is the strongest single signal (42.7% versus 2.8%).
2. **The first year is the danger zone.** Nearly half of new customers leave within 12 months.
3. **Friction signals risk.** Manual payment (electronic check) is associated with churn far more than automatic payment.
4. **Premium service is not enough.** Fiber customers churn most, which points to service quality, price or competition rather than demand. This needs investigation by the product and network teams.
5. **Support matters.** Customers without tech support or online security churn much more.
6. `gender` and `PhoneService` show no meaningful association with churn.

**Interpretation limits.** These are associations, not proven causes. A customer on a month-to-month plan may be leaving for reasons unrelated to the contract. Also, `MonthlyCharges` looks "protective" inside the model even though churners pay more; this happens because price overlaps with the service columns (explained in the notebook) and must not be read as "higher prices reduce churn".

---

## 4. Analytical solution and what it delivers

### 4.1 Model

A tuned Logistic Regression (`C=3`) inside a leak-free scikit-learn pipeline. It was chosen over Gradient Boosting because the boosted model was better by only about 0.004 CV ROC-AUC, below the margin of 0.005 set in advance for giving up interpretability.

| Measure (held-out test set, 1,409 customers) | Value |
|---|---|
| ROC-AUC | 0.842 |
| PR-AUC (random = 0.27) | 0.631 |
| Recall at contact threshold 0.30 | 75% |
| Precision at contact threshold 0.30 | 52% |

### 4.2 Business-ready outputs

**Risk tiers** (out-of-fold on training data, `reports/risk_tiers.csv`):

| Tier | Probability | Share of customers | Actual churn rate | Share of all churners |
|---|---|---|---|---|
| High | >= 0.50 | 22.2% | 65.5% | 54.7% |
| Medium | 0.30 - 0.50 | 15.6% | 38.2% | 22.4% |
| Low | < 0.30 | 62.3% | 9.7% | 22.9% |

**Targeting efficiency** (test set, `reports/lift_table.csv`):

| Customers contacted (riskiest first) | Churners reached | Lift vs random |
|---|---|---|
| 10% | 27.5% | 2.75x |
| 20% | 49.7% | 2.49x |
| 30% | 65.0% | 2.17x |
| 50% | 87.2% | 1.74x |

**Trustworthy probabilities.** Out-of-fold calibration is close to the diagonal and the Brier score is 0.135 versus 0.195 for a no-skill model (`reports/figures/15_calibration_curve.png`). Probabilities can be communicated as risk percentages, after re-checking on new data.

**Explanations.** `src/predict.py` returns the top three risk drivers for every customer (for example *tenure; InternetService; Contract*), so an agent knows what to talk about.

### 4.3 What did not help

Service counts, price per add-on slot, auto-pay and new-customer flags, and dropping the redundant `TotalCharges` were tested with 5-fold CV on the training set. Every variant scored 0.8461-0.8464 ROC-AUC versus 0.8464 for the current features (`reports/feature_experiment.csv`). The model is therefore not limited by feature construction; it is limited by the information in the data.

---

## 5. Feasibility assessment

### 5.1 Technical feasibility: **High**

* Data needed is ordinary billing and contract data that any operator holds.
* Training takes seconds; scoring thousands of customers is instantaneous (`python src/predict.py`).
* The model is a simple, auditable linear model with stable, explainable behaviour.

### 5.2 Financial feasibility: **Conditional**

Test set, base assumptions: offer cost 1, value of a lost customer 8, offer success rate 40% (`reports/business_scenario.csv`).

| Strategy | Customers contacted | Churners reached | Total cost | Saving vs do nothing |
|---|---|---|---|---|
| Do nothing | 0 | 0 | 2,992.0 | 0 |
| Contact everyone | 1,409 | 374 | 3,204.2 | **-212.2** |
| Contact a random 30% | 425 | 120 | 3,033.0 | -41.0 |
| **Model: contact p >= 0.30** | 544 | 282 | 2,633.6 | **+358.4 (12%)** |

Targeting is what creates value: untargeted outreach destroys it. The break-even rule is to contact a customer when `churn probability >= offer cost / (success rate x customer value)`.

Sensitivity of the saving (% of the do-nothing cost, `reports/business_sensitivity.csv`):

| Value of a lost customer \ Offer success rate | 20% | 30% | 40% | 50% |
|---|---|---|---|---|
| 4x offer cost | 0.0% | 0.0% | 2.1% | 6.9% |
| 8x offer cost | 1.1% | 5.4% | **12.0%** | 20.3% |
| 12x offer cost | 3.6% | 10.8% | 18.9% | 28.2% |

**Reading it.** The programme pays clearly when a lost customer is worth at least about 8 offers and offers save at least 30-40% of contacted churners. In the low corner (cheap customers, weak offers) it is not worth running. The two numbers that decide the business case, the real **value of a retained customer** and the real **offer success rate**, are unknown today and are exactly what the pilot must measure.

### 5.3 Operational feasibility: **Medium**

* Needs a monthly scoring routine, a call/SMS/email workflow and offer inventory.
* Needs a holdout group, which means some high-risk customers deliberately receive no offer during the pilot.
* Agents need the reason codes and a short offer playbook (Section 6).

### 5.4 Data, ethical and legal considerations

* Use the model for retention support, not to withdraw service or worsen terms for high-risk customers.
* Senior citizens churn at 41.7% versus 23.6%. A fairness check across `SeniorCitizen` and `gender` should be completed before launch to confirm offers and contact rules are applied equitably.
* Follow the operator's consent and data-protection rules for any outreach.

**Overall verdict: feasible, with a go/no-go decision to be made after a pilot.** The risk of a pilot is small (a few hundred offers); the information it produces is what the financial case lacks.

---

## 6. Recommendations

| # | Action | Evidence | Owner |
|---|---|---|---|
| 1 | Move month-to-month customers to 1-2 year plans with a discount or perk | 42.7% vs 2.8% churn | Marketing |
| 2 | Structured onboarding and check-ins during months 1-6 | 47.4% first-year churn | CRM |
| 3 | Incentive to switch from electronic check to auto-pay | 45.3% vs 16.0% | Billing |
| 4 | Investigate fiber service quality, pricing and competition | 41.9% churn | Product / Network |
| 5 | Offer tech support and online security as a free trial to at-risk accounts | 41.6% churn without tech support | Product |
| 6 | Score customers monthly; contact High tier first, Medium if budget allows | Section 4.2 | Analytics + CRM |
| 7 | Validate everything with an A/B test and re-tune the threshold with real numbers | Section 5.2 | Analytics + Finance |

**Suggested tier playbook**

| Tier | Treatment |
|---|---|
| High (>= 0.50) | Personal outreach, contract-upgrade offer, tailored to the top risk drivers |
| Medium (0.30-0.50) | Low-cost automated offer (email/SMS), auto-pay or support-bundle incentive |
| Low (< 0.30) | No offer; standard service; monitor |

---

## 7. Risks and limitations

| Risk | Impact | Mitigation |
|---|---|---|
| Single snapshot with no dates | No drift check, random split may be optimistic | Time-based validation with real history |
| Financial inputs are assumptions | Business case may change sign | Replace with real value, cost and success rate; use the sensitivity table |
| Association is not causation | Offers may not work as hoped | A/B test with a holdout group |
| Model ceiling (AUC about 0.85) | Many churners remain unpredictable | Collect usage, complaint and support-call data |
| Public sample data | Results may not transfer to the operator's market | Retrain and revalidate on company data |
| Fairness | Unequal treatment of groups | Fairness check before launch |

---

## 8. Roadmap

| Phase | Activities | Exit criterion |
|---|---|---|
| Days 0-30 | Replace assumptions with real figures; fairness check; connect the scoring script to the customer base | Approved offer cost, value and success-rate assumptions |
| Days 30-90 | Pilot: contact the High tier, hold out a random control group | Measured uplift and cost per saved customer |
| Days 90-180 | Scale if the pilot beats break-even; add value-weighted targeting and richer data | Positive net saving at scale |
| Ongoing | Monthly scoring, quarterly retraining, drift and calibration monitoring | Stable performance versus baseline |

**KPIs to track:** churn rate by tier, uplift (treated minus control), cost per saved customer, offer take-up, precision and recall of the contact list.

---

## 9. Reproducibility

| Item | Command / file |
|---|---|
| Full analysis | `notebooks/01_customer_churn_prediction_explainable_ai.ipynb` |
| Rebuild model, calibration, tiers, lift | `python src/train.py` |
| Business scenario and sensitivity | `python src/business_analysis.py` |
| Feature-engineering test | `python src/feature_experiment.py` |
| Score customers with reasons | `python src/predict.py --input <csv> --output <csv>` |
