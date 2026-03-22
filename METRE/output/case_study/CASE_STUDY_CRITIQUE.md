# Critique: Why This Case Study Does Not Prove the Tool's Value

## 1. Cherry-Picked Cases

The 10 patients were **explicitly selected** to make the model look good: we chose the deaths with the highest predicted risk and the survivors with the lowest predicted risk. That is the definition of selection bias. Of course the model "separates" them—we curated the sample for that outcome.

A fair test would be a random sample of normal-at-entry patients, or all of them. We did not do that.

## 2. "Similar" Is a Stretch

Patient A (65, surgical same-day) and Patient B (62, emergency) differ in admission type: elective surgical vs. emergency. Different acuity, different reasons for ICU. (Patient B is actually younger, so age does not explain the model's prediction—but admission context might. The model may be reflecting acuity or patterns in labs/interventions that correlate with emergency presentation.) We did not match on confounders. The comparison is suggestive, not rigorous.

## 3. "Normal" Is Too Broad

Both patients had vitals "within normal limits," but the ranges are wide. Patient B had respiratory rate 20–27 (upper end); Patient A had 10–26. Patient B's RR was consistently higher. That could be an early stress signal. We called both "normal" and then claimed the model found something clinicians missed—but we may have ignored signals that were already there.

## 4. The Model's Discrimination Is Modest

Mean predicted risk: **19%** for deaths vs. **6.7%** for survivors. That is a 12-point gap, not a clean separation. Many deaths get low scores; many survivors get higher scores. We highlighted the best examples and ignored the rest. In the full cohort, the model's AUC is ~0.81—good but not perfect. It misses a lot.

## 5. The 85-Year-Old "Observation" Patient

An 85-year-old in the ICU has elevated mortality regardless of vitals. Predicting high risk for her is not surprising. The model may be driven by age-related patterns in labs, interventions, or physiology—not by subtle early deterioration. "Sicker than she looks" might just mean "old." We did not show that the model adds information beyond what clinicians already know.

## 6. Unclear Actionability

What would the ICU actually do differently? "Prioritize rounds"—they already round on everyone. "Consider escalation"—they are already in the ICU. "Start goals-of-care sooner"—maybe, but we have no evidence that earlier conversations change outcomes or that the model would trigger them appropriately. The benefits are asserted, not demonstrated.

## 7. The Model Sees More Than "First 8 Hours"

We defined "normal at entry" using the first 8 hours. The model uses the **first 24 hours**. So it sees hours 8–24 as well. If deterioration begins in that window, the model is using information from the deterioration period—not purely "early" prediction. We are not showing that the model predicts from truly stable baseline data alone.

## 8. No Evidence of Improved Outcomes

We showed that the model assigns different scores to different patients. We did not show that acting on those scores improves survival, length of stay, or any other outcome. Prediction is not the same as impact. Without a prospective trial, the case for "value" is speculative.

## 9. Small Sample, No Statistics

Ten patients. No confidence intervals. No p-values. No comparison to clinician judgment or simpler rules (e.g., age > 80). A handful of anecdotes is not evidence.

## 10. We Hid the Failures

We did not show:
- High-risk predictions who survived (false alarms)
- Low-risk predictions who died (misses)
- The full distribution of predictions

A complete picture would include these. We presented only the cases that support the narrative.

---

## Bottom Line

The case study tells a story that fits the thesis. It does not prove that the tool adds value. To do that, we would need: matched comparisons, prospective evaluation, outcome data, and an honest account of the model's errors. This document does not provide that.
