# Poster Speaker Notes

Companion notes for the STAI-X 2026 poster session. Organized to match the
poster's block order left-to-right, top-to-bottom. Each section has: what's
on the poster (so you can point at it), a talking-point script for a
30-60 second walkthrough, and "if asked" material for when someone stops and
digs in. Adapted from the presentation slides (`report/slides.tex`), updated
from the 2-model (GPT-4o-mini/Llama) phase to the full 7-model, 3-family
result set the poster reports.

---

## Motivation

**On the poster:** ECE formula, the fairness-audit analogy, three research
questions.

**Talking points:**
"LLMs are deployed across law, medicine, math — and users treat stated
confidence as a reliability signal. But calibration is almost always reported
as one aggregate number, ECE, averaged over an entire benchmark. That hides
*where* miscalibration concentrates. The analogy I use: we'd never accept a
single accuracy number across demographic groups as a fairness audit — the
same logic applies here. A single ECE is not a calibration audit."

**If asked: how does the ECE formula work?**
Bin questions by confidence value into equal-width bins (we use 20). For each
bin, compare mean empirical accuracy to mean stated confidence; weight by bin
size. It's agnostic to how confidence was elicited — here it's renormalized
softmax over the answer tokens.

**If asked: why does bin count matter?**
Wider bins average out real variation in confidence within the bin, which can
mask miscalibration. This is exactly why we ran the bin-count sensitivity
check (see Robustness, folded into "Magnitude, Not the Binary Count") —
rankings are stable across $b \in \{10,15,20,25,30\}$.

---

## Data & Methods

**On the poster:** MMLU stats, confidence formula, model list, three-level
mixed-model equation.

**Talking points:**
"MMLU has 57 subjects — things like clinical medicine, formal logic, high
school math — nested in 4 broad domains. That nested structure is why a
multilevel model is the right tool. Confidence is the renormalized softmax
probability the model assigns to its top answer token. The primary outcome
is the *signed* calibration gap — confidence minus correctness — not just
ECE, because the sign tells us the *direction* of miscalibration (positive =
overconfident)."

**If asked: gap vs. ECE — when do we use which?**
- **Gap** (question-level, signed): the outcome for the multilevel
  regression; BH t-tests test whether a subject's mean gap is 0.
- **ECE** (subject-level, unsigned): used to rank subjects, drives the
  reliability curves, and is what the cross-model Spearman correlation
  compares.
- They're numerically close here because every subject is uniformly
  overconfident (when conf > acc everywhere, ECE ≈ mean gap), but they play
  different analytical roles — gap supports signed inference, ECE is the
  standard scalar for ranking/comparison.

**If asked: why REML, not standard MLE, for the variance components?**
MLE treats the fixed-effect estimates as known with certainty when computing
residuals, which systematically *underestimates* variance components —
analogous to dividing by $n$ instead of $n-p$. REML accounts for the degrees
of freedom spent estimating the fixed effects and gives unbiased variance
estimates. This matters a lot here because $\sigma^2_\text{subject}$ and the
ICC are our primary inferential targets, not nuisance parameters — MLE would
push the ICC toward zero and understate how much subject membership explains.

**If asked: why fixed effects for question features but random effects for
subject/domain?**
At the question level we have a theory: longer questions are harder, higher
entropy means more model uncertainty, negation is known to trip models up —
you can name the mechanism. At the subject/domain level we don't have a
theory of *which* subject-level features drive miscalibration (abstractness?
rarity in pretraining data?) — so we let the data estimate how much subjects
differ without committing to why. It's also more efficient: fitting 57
subject fixed effects would burn degrees of freedom on subjects with as few
as 100 questions.

---

## Structure Exists, But Is Modest

**On the poster:** ICC table (7 models), text noting ICC spans ~5× across
models but domain variance is ~0.

**Talking points:**
"Before showing which subjects are worst, it's worth being upfront: most
variance in the calibration gap — 92 to 98.5% — is at the question level, not
the subject level. Between-subject ICC ranges from about 1.7% (Qwen-1.5B) to
7.9% (GPT-4o-mini). That sounds small. But — as the next panel shows — it's
enough to organize a 3.8–13× range in how miscalibrated different subjects
are. Small ICC and large practical range are compatible facts, not a
contradiction."

**If asked: what does ICC actually mean?**
$\text{ICC}_\text{subject} = \sigma^2_\text{subject} / (\sigma^2_\text{domain}
+ \sigma^2_\text{subject} + \sigma^2_\text{residual})$. Intuition: if you drew
two questions at random from the *same* subject, ICC is how correlated their
calibration gaps would be, purely from shared subject membership. ICC = 0
means subject tells you nothing; ICC = 1 means every question in a subject
has an identical gap.

**If asked: why is domain variance exactly zero?**
With only 4 domain groups, the model has very low power to separate
domain-level variance from subject-level variance — the subject random
effects absorb whatever between-domain structure exists. This is a
structural limitation of MMLU's 4-domain design, not a modeling failure. Be
upfront about it if pressed.

---

## Where Calibration Fails

**On the poster:** universal-overconfidence claim + intercept range, the
worst/best subject lists, GPT-4o-mini reliability curves (top-5 worst,
bottom-5 best).

**Talking points:**
"Every one of the seven models is overconfident on average — the intercept
is positive for all of them, 0.18 to 0.31. That's consistent with prior work
on RLHF-tuned models. But it's not uniform: `virology` and
`professional_law` are in the top-10 worst-calibrated for *every* model above
1B parameters; a cluster of college-level STEM subjects — abstract algebra,
college math, machine learning — shows up in the top-5 for four of six.
Symmetrically, `marketing` and `high_school_psychology` are consistently
best. The organizing axis is specialization level: high-school subjects are
reliably better calibrated than their college-level counterparts."

**If asked: what do "below the diagonal" and "above the diagonal" mean on the
reliability curves?**
Below the diagonal = overconfident (model says 90%, right 50% of the time —
users trust it more than they should). Above = underconfident (hedges more
than necessary). We find universal overconfidence, consistent with RLHF
pushing probability mass toward the chosen answer for confident-sounding
responses. Overconfidence is the worse deployment failure — underconfidence
prompts users to double-check; overconfidence in legal/medical settings
actively misleads.

**If asked: the curves look like they only have data on the right side —
why?**
The confidence distribution is heavily right-skewed (median confidence
across all questions is essentially 1.0). The reliability curves are honest
about this — solid isotonic lines only extend across the *observed* support;
we don't extrapolate into empty low-confidence bins. That's also why we
picked isotonic (holds the last observed value flat) over a kernel smoother
(which can produce misleading smoothing artifacts over empty regions).

---

## Magnitude, Not the Binary Count

**On the poster:** $H_0$/$H_1$ equation, BH-FDR description, the 3.8–13.0×
claim, robustness footnote.

**Talking points:**
"For each of the 57 subjects we run a one-sample t-test: is the mean gap
zero? With Benjamini-Hochberg correcting across all 57 tests at α = 0.05.
The headline number — 55 to 57 out of 57 subjects rejected, depending on
model — sounds impressive but isn't the interesting part. With 14,000+
questions and a model that's already globally overconfident, you have enough
statistical power to reject almost anything nonzero. It's like testing
whether 57 cities are warmer than absolute zero — technically true, not
informative. The actual finding is the *magnitude*: best- to worst-calibrated
subject gap ranges span 3.8× to 13.0× across the six larger models, and the
*same* subjects sit at each extreme every time."

**If asked: why doesn't the near-universal rejection undermine the finding?**
It's the opposite problem from what BH is built to catch. BH controls false
positives across simultaneous tests — but here the issue isn't false
positives, it's that all the true effects are real yet some are scientifically
trivial (e.g., the best subject, `marketing`, still has gap ≈ 0.037 — non-zero
but small). BH has nothing to correct when p-values are near zero for
structural reasons (a large global intercept), not because 57 hypotheses are
being tested. The magnitude range is the number that actually distinguishes
"barely miscalibrated" from "off by 40+ points."

**If asked: how robust is this to methodological choices?**
Two checks: (1) ECE subject rankings are stable across bin counts 10–30
(Spearman ≥ 0.97 vs. our 20-bin baseline) and agree with a bin-free, strictly
proper scoring rule, the Negative Log Calibration Score (Spearman ≥ 0.79).
(2) Two- and three-level ICC estimates are numerically identical, so the
domain level isn't secretly doing work. Neither the ranking nor the variance
story is an artifact of a specific modeling choice.

---

## Stable Across Models and Families

**On the poster:** 7×7 Spearman correlation heatmap.

**Talking points:**
"This is the load-bearing result for the whole poster. Among the six models
above 1B parameters, all 15 pairwise Spearman correlations of subject-level
ECE fall between 0.80 and 0.94. Cross-family pairs — GPT-4o-mini vs.
Qwen-7B or 14B — are just as strong (0.92, 0.91) as the best within-family
pair (Qwen-7B vs. 14B at 0.94). That means the pattern isn't a quirk of one
architecture or training pipeline; the same subjects are hard *regardless of
who trained the model or how big it is*. Qwen-0.5B is the one exception,
correlating only 0.13–0.20 with everyone else — but that's explained by a
10.6× positional bias toward answer D, not genuine subject-level knowledge
gaps. We read that as a floor below which our elicitation protocol breaks
down, not a counterexample to the stability claim."

**If asked: what can and can't you conclude from a Spearman ρ this high?**
*Can conclude:* subject rankings are highly consistent across models; this
is a property of the knowledge domains, not of a specific training pipeline —
model-specific retraining is unlikely to fix it.
*Cannot conclude:* that absolute ECE values match (Spearman only compares
orderings, not magnitudes); or that this generalizes indefinitely beyond the
7 models/3 families tested here — that's suggestive at this scale, not
proof for arbitrary future models.

**If asked: why does this matter practically?**
It's the precondition that makes subject-aware recalibration worth building.
If the hard subjects were different for every model, you'd need to
re-diagnose from scratch each time. Because they're stable, a correction (or
even just a warning list) built on one model transfers to the next — see the
"Why Subject-Level" callout.

---

## What Predicts the Gap

**On the poster:** forest plot of fixed effects across all 7 models
(intercept, word count, max option length, negation, entropy).

**Talking points:**
"These are seven separately-fit models — not one joint model — so read the
columns as independent estimates. Word count positively predicts
overconfidence for GPT-4o-mini and the larger Qwens; negation and option
length are mostly inert everywhere. The one pattern worth dwelling on is
entropy, which flips sign across the Qwen scaling sweep — that's the next
panel."

**If asked: why would word count predict the gap at all?**
It's likely partly a proxy for subject identity rather than a pure
question-level effect. Subjects that differ in baseline calibration (higher
ICC models especially) also tend to differ systematically in question
length/complexity — so word count picks up some of that between-subject
signal. A cleaner test would group-mean-center word count within subject to
isolate the pure within-subject effect; we didn't do that here, so treat the
fixed effects as associations, not fully isolated causal question-level
effects. The primary findings (ICC, the magnitude range, cross-model ρ)
don't depend on these fixed effects at all.

---

## The Gap Grows With Scale

**On the poster:** entropy coefficient sign progression across Qwen scale,
confidence-vs-accuracy-by-entropy-quartile panels (all 7 models).

**Talking points:**
"Naive prediction: higher entropy over the four answer options should
*shrink* the gap, because top-1 confidence is mechanically lower when
probability mass is spread out. Instead, within the matched-precision Qwen
sweep — 1.5B, 3B, 7B, 14B — the entropy coefficient grows monotonically from
-0.004 to +0.040. At 7B and 14B, confidence stays near saturation across the
first three entropy quartiles while accuracy keeps dropping — so the gap
widens specifically on the model's *hardest* items. Confidence does fall with
entropy, just slower than accuracy falls. That's consistent with RLHF-style
confidence saturation reported elsewhere — bigger, more heavily-tuned models
sound equally sure of themselves whether or not they're right."

**If asked: is this really about scale, or about something else correlated
with scale?**
We control for this by using a matched-precision sweep within one family
(Qwen 1.5B/3B/7B/14B, same quantization/precision), so it isn't an artifact of
comparing different training recipes or numeric precision across families —
it's a within-family scale effect. The between-subject ICC also jumps from
1.5B to 3B (0.017 → 0.051) and is flat after that, which lines up with the
same "something changes around 3B" story.

**If asked: does this contradict the "task-driven, not model-driven"
claim?**
No — it's a different axis. Cross-model stability says *which* subjects are
hardest is consistent across models. This says the *severity* of the
confidence/accuracy mismatch on hard items grows with scale within a family.
Both can be true: the ranking is stable, but larger models are more
confidently wrong on the specific items that are already hard for everyone.

---

## Why Subject-Level, Not Model-Level (the callout)

**On the poster:** red alert block — the temperature-scaling argument, the
pre-deployment-checklist upshot.

**Talking points — this is the "so what," don't rush it:**
"The standard post-hoc fix for miscalibration is temperature scaling — one
scalar $T$ that rescales every logit uniformly. The problem: a single $T$
tuned to fix the worst subjects (professional_law, virology — gap up to
0.487) would badly *overcorrect* a well-calibrated subject like marketing
(gap 0.037), pushing it into underconfidence. There is no single temperature
that works for subjects whose gaps differ by 13×. And because the same
subjects are hard across every family and scale we tested, this isn't a
quirk you can train away — a better future model would likely reproduce the
same ranking of hard vs. easy subjects. The fix has to operate at the subject
level: a separate temperature (or correction) per subject. Practically, the
recurring worst-calibrated subjects double as a portable pre-deployment
checklist — you can flag likely miscalibration on a brand-new model without
re-running the full 57-subject audit."

**If asked: why can't retraining/better RLHF just fix this?**
Because the heterogeneity is task-driven (ρ ∈ [0.80, 0.94] across families
and scales), the hard subjects aren't an artifact of one model's training
pipeline — they reflect something about the *content itself* (abstractness,
rarity, ambiguity — exactly what's still unresolved, see Takeaways). A model
retrained from scratch would likely still find `virology` harder to
calibrate than `marketing`.

---

## Takeaways

**On the poster:** three bolded bullets — unit of analysis, task-driven,
open question.

**Talking points (closing, ~20-30 sec):**
"Three things to leave with: first, the right unit of analysis for
calibration is the subject, not the model — a single ECE hides a 13×
range that matters practically. Second, miscalibration is task-driven:
it's stable across three model families and six scales, so you shouldn't
expect a better model to fix it on its own. Third, we're honest about what's
still open — we can say *which* subjects are hardest, but not fully *why*.
That's the natural next step."

**If pushed on the open question:**
Candidate explanations we haven't disentangled: abstractness/reasoning depth
required, rarity of the topic in pretraining data, ambiguity in how MMLU
phrases the question, or some interaction of these. Distinguishing them would
need controlled interventions (e.g., varying question phrasing for matched
content), which is out of scope here.

---

## References

No dedicated talking points — cite on request. If someone asks about a
specific claim (RLHF-induced saturation, QLoRA precision loss, etc.), the
supporting citation is listed on the poster next to or near that claim.
