# Case Study: Why Early Risk Prediction Matters in the ICU

## The Problem

In the ICU, clinicians see dozens of patients each day. Many arrive with vitals that look stable—heart rate, blood pressure, oxygen saturation all within normal ranges. The question: *Which of these "stable" patients will deteriorate?* Without a way to distinguish them, everyone gets similar attention. Resources spread thin. The ones who need extra vigilance may not get it until it's too late.

## The Setup

We identified 10 ICU stays where patients looked **similar and normal at entry**: first 8 hours of vitals (heart rate 60–100, SBP 90–140, respiratory rate 12–22, SpO2 95–100%, temperature 36–37.5°C). No obvious red flags. Five of these patients died. Five survived.

Our model uses only the first 24 hours of ICU data—vitals, labs, interventions—to predict in-hospital mortality. No demographics. No chart review. Just the numbers.

## The Story: Two Patients, Same First Day

### Patient A (Survived)

- **Age 65, male.** Surgical same-day admission. ICU stay: 1.2 days.
- **First 8 hours:** Heart rate 60–88, SBP 92–126, SpO2 98–100%, respiratory rate 10–26. All within normal limits.
- **Model prediction:** 1.3% mortality risk.
- **Outcome:** Discharged to home health care.

### Patient B (Died)

- **Age 62, male.** Emergency admission. ICU stay: 1.9 days.
- **First 8 hours:** Heart rate 81, SBP 138, SpO2 100%, respiratory rate 17. Also within normal limits.
- **Model prediction:** 70% mortality risk (LR), 47% (RF).
- **Outcome:** Died in hospital.

At the bedside, both looked stable in the first day. Patient B was actually younger than Patient A. Routine vitals did not separate them. The model did.

## What the ICU Gains

**Without the tool:** Two patients with normal-looking vitals. Same triage. Same level of monitoring. No way to know which one will deteriorate.

**With the tool:** Patient B gets flagged as high risk from day one. The ICU can:
- Prioritize rounds and nursing attention
- Consider earlier escalation (e.g., step-down vs. floor)
- Alert the team before vitals begin to drop
- Start family and goals-of-care conversations sooner

## Another Angle: The "Observation" Patient

One of the deaths was an **85-year-old woman** admitted for *observation*—suggesting the team initially thought she might not need intensive care. Her first 8 hours: HR 87, SBP 102, SpO2 99%, all normal. The model assigned **90% mortality risk**. She died after 8 days in the ICU.

The tool would have signaled: *This patient is sicker than she looks.* That could have changed triage, monitoring intensity, or how early the team prepared for a poor outcome.

## The Numbers

Among the 3,376 ICU stays with "normal" baseline vitals in our cohort:
- **185 died**, 3,191 survived
- The model's mean predicted risk: **19%** for those who died vs. **6.7%** for those who survived
- It separates outcomes using patterns in early data that are hard for clinicians to see in real time

## Takeaway

Stable vitals at admission do not mean low risk. Our tool uses the first 24 hours of ICU data to identify patients who look fine but are at high risk of death—before they deteriorate. That gives the ICU a chance to act earlier.
