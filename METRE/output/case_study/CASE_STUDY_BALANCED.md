# Case Study: Early Risk Prediction in the ICU—What We Can and Cannot Say

## The Problem (This Part Is Real)

In the ICU, many patients arrive with vitals that look stable. Clinicians cannot easily tell which of these "stable" patients will deteriorate. That uncertainty is genuine. The question—*can we do better?*—is worth asking.

## What We Did

We built a model that uses the first 24 hours of ICU data (vitals, labs, interventions) to predict in-hospital mortality. We then looked at patients whose first 8 hours of vitals fell within physiologic norms—heart rate 60–100, SBP 90–140, respiratory rate 12–22, SpO2 95–100%, temperature 36–37.5°C. Among 3,376 such stays, 185 died and 3,191 survived. The model's mean predicted risk was 19% for those who died vs. 6.7% for those who survived. So it does separate outcomes to some degree, even among patients who look fine at entry.

## A Panel of Five Patients—Illustrative, Not Proof

We picked five to illustrate the idea. All had first 8 hours of vitals within physiologic norms (HR 60–100, SBP 90–140, RR 12–22, SpO2 95–100%, Temp 36–37.5°C). First, the baseline picture—what clinicians see at the bedside:

| Age | Admission type | First 8h vitals (mean) |
|-----|----------------|------------------------|
| 57, M | Urgent | HR 89, SBP 100, RR 20, SpO2 99.2%, Temp 37.2°C |
| 58, M | Emergency | HR 60, SBP 115, RR 17, SpO2 98.6%, Temp 36.5°C |
| 62, M | Emergency | HR 81, SBP 138, RR 17, SpO2 100%, Temp 36.9°C |
| 63, M | Surgical same-day | HR 77, SBP 113, RR 15, SpO2 99%, Temp 36.4°C |
| 69, F | Emergency | HR 90, SBP 116, RR 15, SpO2 99.6%, Temp 37.1°C |

Adding model predictions and outcomes. *Model risk* is the predicted in-hospital mortality probability after 24 hours of ICU observation (LR = logistic regression, RF = random forest):

| Age | Admission type | First 8h vitals (mean) | Model risk (LR / RF) | Outcome |
|-----|----------------|------------------------|----------------------|---------|
| 57, M | Urgent | HR 89, SBP 100, RR 20, SpO2 99.2%, Temp 37.2°C | 2.3% / 0% | Survived → Home health |
| 58, M | Emergency | HR 60, SBP 115, RR 17, SpO2 98.6%, Temp 36.5°C | 13% / 0% | Survived → Rehab |
| 62, M | Emergency | HR 81, SBP 138, RR 17, SpO2 100%, Temp 36.9°C | 70% / 47% | Died |
| 63, M | Surgical same-day | HR 77, SBP 113, RR 15, SpO2 99%, Temp 36.4°C | 6% / 0% | Survived → Home health |
| 69, F | Emergency | HR 90, SBP 116, RR 15, SpO2 99.6%, Temp 37.1°C | 60% / 50% | Died |

We biased the panel toward younger patients (all under 70) so the model is not simply reflecting age. We also included non-emergency admissions (surgical, urgent) where possible. Admission type alone is not always the most helpful predictor—the 62-year-old and 69-year-old emergency patients died with 47–70% risk, while the 58-year-old emergency patient survived with 13% risk. Same admission type, different outcomes. The model may be picking up on acuity or patterns in labs and interventions that we did not show. A fair comparison would match on confounders. We did not do that. So this is an illustration of possibility, not evidence of superiority over clinical judgment.

## What We Are Not Claiming

- **We did not prove** that the tool improves outcomes. We showed it assigns different scores; we did not show that acting on those scores changes survival or length of stay.
- **We did not use** a random sample. We selected cases to show the model at its best. That selection bias means we overstate how well it performs in practice.
- **We did not show** that the model adds information beyond what clinicians already have. Age, admission type, and early labs may explain much of the signal. A comparison to clinician judgment or simple rules (e.g., age + admission type) would be needed.
- **We did not demonstrate** clear actionability. What would the ICU do differently? We suggested possibilities—earlier goals-of-care, different triage—but we have no data that those actions would help.

## What We Can Reasonably Say

1. **The model discriminates.** Among patients with normal-looking early vitals, mean predicted risk is higher for those who die than for those who survive. The effect is modest (19% vs. 6.7%), not dramatic, but it is there.

2. **The model uses only structured data.** No chart review, no free text. That makes it scalable and auditable. Whether it adds value on top of clinician judgment is an open question.

3. **Some high-risk patients look stable at first.** Among the panel above, the 62- and 69-year-olds had normal early vitals and died; the model assigned them 47–70% risk. Whether it is capturing something beyond admission context is unclear, but the pattern exists.

4. **The tool is not ready for deployment.** It would need prospective validation, outcome studies, and a clear plan for how clinicians would use it. This case study is a step toward that conversation, not the end of it.

## The Honest Takeaway

Stable vitals at admission do not guarantee low risk. Our model suggests that, even among patients who look fine early on, there are patterns in the first 24 hours that correlate with mortality. We have not shown that the tool improves care or that it outperforms simpler approaches. What we have shown is that the question is worth pursuing—and that any future claims of value will need to be backed by the kind of rigorous evaluation this case study does not provide.
