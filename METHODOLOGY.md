# Methodology: what was changed, and why

This file explains every change made to the labelling pipeline of this repository, the problem
behind each change, and the published research each solution relies on. It is written so it can be
read on its own (for example by a supervisor) and reused for the methodology section of the paper.

**Research goal.** Which programming tasks does real Pharo code perform, and how often?

**Constraint.** About 280,000 methods in the study corpus (135 repositories). Labelling all of them
with large commercial LLMs costs too much, so the study uses statistical sampling, local open-weight
models, saturation, and validation by Pharo experts.

**Research questions the pipeline answers.**

| RQ | Question | Where the answer comes from |
|---|---|---|
| RQ1 | What task categories exist in Pharo code, and how common is each? | Final taxonomy with shares corrected by the experts (step 5) |
| RQ2 | How accurate are local LLMs at labelling Pharo methods, compared with Pharo experts? | Stratified accuracy of each model and of the majority vote (step 5) |
| RQ3 | Does agreement between models predict correctness? | Accuracy per agreement level, 3/3 vs 2/3 vs none (step 5) |

---

## 1. The pipeline, before and after

**Before**

```
mine → filter → one model labels a sample (open coding) until saturation
     → a second model re-labels 60 methods (Cohen's kappa, only reported)
     → expert files exported (never read back)
```

**After**

```
1. mine       GitHub → filter 1 (is this real Pharo code?)
2. filter     filter 2 → 135 repositories, ~280,000 methods
3. label      INDUCTIVE: qwen3.8 names the task of each sampled method until saturation
              → codebook_draft.csv: the categories still at ≥ 0.5% at the end, each with a
                definition, "use when" and "do not use when" written by the model
   [you]      copy it to codebook.csv and consolidate it (merge, rename, define)
4. classify   DEDUCTIVE: qwen3.8 + gemma4 + mistral-small3.2 classify a NEW random sample (384)
              → Krippendorff's α, Gwet's AC1, Fleiss' κ, prompt-sensitivity check
              → α < 0.667: stop, NOT_RELIABLE.md; a revised codebook or model list is a new
                revision, measured on a new sample disjoint from the earlier ones
              → α ≥ 0.667: 200 methods for the experts, stratified by agreement level
5. validate   two experts check the 200 methods in method_review.csv, each alone (the
              models' label filled in, changed when wrong)
              → accuracy of each model (95% CI), experts' agreement (human baseline),
                precision/recall per category, share of each task corrected by the experts (PPI)
6. report     results/REPORT.md
```

Commands: `./run.sh` runs everything; each step alone: `./run.sh label | classify | validate | report`.

---

## 2. The problems and their solutions

### Problem 1: the labelling models were code-completion models

**What was wrong.** The default labellers were `pharo-llm/Qwen2.5-Coder-SFT` (0.5B to 7B) and the
`-PT` LoRA adapters. These models were fine-tuned to *write Pharo code*. Labelling means *describing*
code in English and choosing a category, which needs instruction following. The pipeline worked
around this with a raw few-shot completion prompt. In practice the labels were often wrong or too
specific: a simple setter was labelled "Dependency injection", a Redis call "Redis command".

**Solution.** Use only instruction-tuned general models, asked through Ollama's chat API with
thinking turned off and deterministic decoding (temperature 0, fixed seed). The Pharo models, the
Hugging Face backend and the raw completion mode were removed from the project.

**Model choice.** The machine is an Apple M4 Max with 36 GB of unified memory and little free disk.
The models had to fit in memory, be recent, and come from **different companies** (see Problem 4).

| Model | Company | Size | Role |
|---|---|---|---|
| `qwen3.8:27b` | Alibaba | 18 GB | open coding (step 3), codebook definitions, classifier 1 (step 4) |
| `gemma4:12b` | Google | 8 GB | classifier 2 |
| `mistral-small3.2:24b` | Mistral AI | 15 GB | classifier 3 (replaces `granite4.2:8b` (IBM) after the first run: Problem 14) |

Qwen3.8-27B was chosen as the strongest model that fits: in a recent local comparison it beat
Qwen3.6-27B and Gemma 4 31B on coding, reasoning and document QA, with equal instruction following
(kingy.ai, 2026). That comparison is a single synthetic test suite, not a standard benchmark: this
is why the design does not rely on one model.

**Result.** On methods of the corpus, Qwen3.8 gave general task names where the SFT model gave
domain words ("Accessor" instead of "Dependency injection"). On a hand-written test, all three
models classified 6 methods out of 6 correctly with a codebook, including "Other" for an HTTP call,
at 0.7 to 2 seconds per method.

**Sources.** Baltes et al., *Guidelines for Empirical Studies in SE involving LLMs* (EMSE): use open
models, report versions and configuration.

---

### Problem 2: open codes from different models cannot be compared

**What was wrong.** In open coding, each model invents its own category names. One model writes
"Setter", another "Accessor", for the same task. The old agreement check compared these free names
after an automatic merge by embeddings, so a low kappa could mean "different words", not "different
judgement". The agreement number was not trustworthy.

**Solution: two phases.**

1. **Inductive (step 3):** one model open-codes until saturation. The categories found become a
   codebook draft, with a one-sentence definition per category written by the model
   (`codebook_draft.csv`). The researcher consolidates it into `codebook.csv`: merge duplicates,
   rename, sharpen definitions. "Other" is always added.
2. **Deductive (step 4):** every model classifies methods by choosing **one category of the fixed
   codebook**. Now the models answer the same question with the same vocabulary, and agreement
   measures real agreement.

**Sources.**
- Shah et al., *From Inductive to Deductive: LLMs-Based Qualitative Data Analysis in Requirements
  Engineering* (2025): with a codebook (deductive), GPT-4 reaches substantial agreement with human
  analysts (κ > 0.7); free (inductive) coding performs poorly.
- Wan et al., *TnT-LLM* (KDD 2024): the same two steps at scale: the LLM generates a taxonomy, then
  classifies with it.
- Nakashima et al., *How Well Do LLMs Generate Taxonomies in the SE Domain?* (ASE 2026): LLMs can
  build SE taxonomies comparable to human ones, but they tend to be over-complex, so a human
  consolidation step is needed (here: `codebook.csv`).

---

### Problem 3: testing the codebook on the data it was built from

**What was wrong.** If the models classify the same methods the codebook was built from, the
codebook fits them by construction and agreement looks better than it is (double dipping).

**Solution.** Step 4 draws a **new** random sample (stratified by repository, like step 3, default
384 methods from the sample-size calculator) that excludes every method used in step 3. A test
checks that the two samples never overlap.

**Source.** Standard train/test separation; Cochran (1977), *Sampling Techniques*, for the stratified
draw with proportional allocation.

---

### Problem 4: with two models you cannot tell which one is wrong

**What was wrong.** The old design had one labeller and one check model. When they disagreed, there
was no way to decide, and one model's systematic bias went undetected.

**Solution.** Three models classify every method independently. Each method gets:
- the **majority label** (at least 2 of 3 agree), and
- an **agreement level**: `3/3` (all agree), `2/3`, or `none` (all different).

These levels are used later to choose which methods the experts check (Problem 8) and to test
whether agreement predicts correctness (RQ3).

**Sources.**
- *Wisdom of the LLM Crowd* (arXiv 2602.11962, 2026): treats each LLM as an independent annotator,
  aggregates by majority vote, compares 3- and 5-model votes against human consensus.
- Ahmed, Devanbu, Treude, Pradel, *Can LLMs Replace Manual Annotation of Software Engineering
  Artifacts?* (MSR 2025, ACM SIGSOFT Distinguished Paper): model-model agreement is a signal of
  whether a task suits LLMs.

---

### Problem 5: models make the same mistakes

**What was wrong.** Agreement between models is not proof of correctness. Large models trained on
similar data make *correlated* errors: if they agree, they may agree on the same wrong answer.

**Solution.**
1. The three models come from **three different companies** (Alibaba, Google, Mistral AI), to reduce shared
   training data and architecture.
2. The expert sample includes methods where **all three models agree** (stratum `3/3`), not only
   disagreements: that is where shared errors hide. The report shows the accuracy inside each
   agreement level.

**Sources.**
- Kim, Garg, Peng, Garg, *Correlated Errors in Large Language Models* (ICML 2025): over 350 LLMs;
  when two models are both wrong they agree on the same wrong answer about 60% of the time; larger,
  more accurate models are more correlated, even across providers.
- *Wisdom and Delusion of LLM Ensembles for Code Generation and Repair* (arXiv 2510.21513): consensus
  selection falls into a "popularity trap" that amplifies common wrong answers.

---

### Problem 6: Cohen's kappa is the wrong agreement measure here

**What was wrong.** Cohen's kappa works for two annotators only. It also collapses when one category
dominates (the "kappa paradox": very high agreement, low kappa). In Pharo code a few categories
(accessors, tests) dominate.

**Solution.** Report several measures:

| Measure | Why |
|---|---|
| **Krippendorff's α** (nominal) | any number of annotators, missing answers allowed; the main measure and the gate |
| **Gwet's AC1** | stays stable when one category dominates |
| **Fleiss' κ** | the classic measure for several annotators, for comparison |
| Pairwise Cohen's κ | shows if one model is far from the others |

Thresholds (Krippendorff): α ≥ 0.8 reliable, α ≥ 0.667 acceptable for tentative conclusions.

**Sources.**
- Krippendorff (2004), *Content Analysis*, ch. 11; Krippendorff (2011), *Computing Krippendorff's
  Alpha-Reliability*.
- Fleiss (1971); Gwet (2008), *Computing inter-rater reliability and its variance in the presence of
  high agreement*; Feinstein & Cicchetti (1990), the kappa paradox.
- James, *Counting on Consensus: Selecting the Right Inter-annotator Agreement Metric* (LREC 2026).

**Verification.** The implementation reproduces published values: α = 0.743 on Krippendorff's own
worked example (4 coders, 12 units, missing data), and κ = 0.210 on the classic Fleiss example.
A test also shows the paradox: one dominant category gives Fleiss κ < 0.5 but AC1 > 0.9.

---

### Problem 7: experts' time spent on an unreliable codebook

**What was wrong.** Before, the expert files were always written, whatever the agreement.

**Solution: a reliability gate.** If Krippendorff's α < 0.667, no expert files are written. Instead
`classify/NOT_RELIABLE.md` explains what to do (merge confused categories, sharpen definitions,
replace a model that is far from the others), then step 4 is run again. Above the threshold, the
expert files are produced. The threshold can be changed (`MIN_ALPHA`), and lowering it must be
reported in the paper.

**Sources.** Ahmed et al. (MSR 2025): model-model agreement as the signal that a task suits LLMs.
Imran & Zaman, *OLAF: Towards Robust LLM-Based Annotation Framework in Empirical SE* (WSESE 2026):
treat LLM annotation as a measurement process with reliability and calibration checks.

---

### Problem 8: model answers can depend on the prompt

**What was wrong.** Small changes in prompt wording or order can change LLM labels.

**Solution: a prompt-sensitivity check.** The first model classifies 100 methods a second time with
the codebook categories and the worked examples **shuffled**. Its agreement with itself (Cohen's κ)
is reported. A low value means the definitions are too vague.

**Source.** Camuffo et al., *Variance-Aware LLM Annotation for Strategy Research* (arXiv 2601.02370,
2026): prompt phrasing and model choice can shift outcomes by 12 to 85 percentage points; they
propose a protocol with diagnostics and reporting standards.

---

### Problem 9: the expert sample was small, and partly wasted

**What was wrong.** The experts checked 60 random methods (about ±10% on accuracy) plus 30 methods
where the two models disagreed. The 30 could not be used in the accuracy, because they are not a
random sample: mixing them in would bias the result.

**Solution: a stratified expert sample with known weights.**
- The classified sample is split into strata by agreement level (`3/3`, `2/3`, `none`).
- 200 methods are drawn (about ±5% on accuracy), with **more in the strata where the models
  disagree** (Neyman allocation using guessed error rates 10%, 35%, 60%), and at least 20 per
  stratum, including the unanimous one (Problem 5).
- Each stratum has a known weight (methods in the stratum / methods classified). The accuracy is a
  **stratified estimator**: the weighted mean of the accuracies of the strata. Every expert answer
  counts, and the estimate stays unbiased.
- The strata and weights are stored in `classify/gold_selection.csv`, which the experts never see.

**Sources.**
- Cochran (1977), *Sampling Techniques*, ch. 12 (two-phase sampling for stratification) and ch. 5
  (Neyman allocation).
- Gligorić, Zrnic, Lee, Candès, Jurafsky, *Can Unconfident LLM Annotations Be Used for Confident
  Conclusions?* (NAACL 2025): choose which items humans label from the LLM's uncertainty, keep valid
  conclusions by weighting; more than 25% fewer human annotations needed.

**Verification.** A test checks that the stratified estimate equals the weighted mean (0.75·0.9 +
0.25·0.6), not the raw share of correct answers.

---

### Problem 10: the experts could be influenced, and there was no human baseline

**What was wrong.** Before, the expert file showed the model's label and asked "correct yes/no".
Seeing a label anchors the judgement toward accepting it. With a single expert, there is also no way
to know how hard the task is for humans.

**Solution.**
- **Labels to correct, not to invent (the researcher's choice, `expert_prefill: true`).** In
  `method_review.csv`, `expert_category` already holds the models' majority label (also kept in
  `model_label`); the expert opens the file in a spreadsheet and changes only the wrong rows. Methods
  where the models have no majority come with an empty `expert_category`. The expert also gets
  `taxonomy.png` (the categories and their shares) and `codebook_review.csv` to validate the
  taxonomy. This is the least work for the experts, and it has a known cost: pre-annotation speeds
  annotation up but introduces biases (Fort & Sagot, LAW 2010), and annotators lean toward the label
  they edit, which also raises their agreement with each other (Berzak et al., EMNLP 2016). A row
  the expert never looked at cannot be told from a row they confirmed. So the accuracy of step 5 is
  an upper bound, and the experts' α is not an independent human baseline. Step 5 reports how many
  labels each expert changed (`changed_model_label` in `validation.json`). With
  `expert_prefill: false` the file is blind and the accuracy is unbiased.
- **Two experts, working alone.** Their agreement (Krippendorff's α, Cohen's κ) is the **human
  baseline**.
- Where the experts disagree, they discuss and put the agreed answer in `returned/consensus/`.
  Methods still disputed are left out and counted.
- The experts also review the codebook (keep, rename, merge, better definition). Decisions are taken
  by majority; ties go to `validated/to_discuss.csv` and are not applied.
- The comparison that matters: a model can stand in for a human rater when its agreement with the
  reference labels is close to the experts' agreement with each other.

**Source.** Ahmed et al. (MSR 2025): LLMs can replace one human rater when model-human agreement is
close to human-human agreement; they reduce, not replace, human effort.

---

### Problem 11: shares computed from LLM labels are biased

**What was wrong.** This is the most important statistical problem. Saying "accessors are 16% of
Pharo methods" from LLM labels gives a biased number: the model's errors are not random, so even a
model that is 80-90% accurate shifts the shares, and the confidence interval is wrong.

**Solution: prediction-powered inference (PPI).** For each category:

```
corrected share = share given by the models on the 384 classified methods
                + weighted mean, over the expert-checked methods, of
                  (expert says this category − models say this category)
```

The second term (the "rectifier") measures and removes the models' bias. The 95% interval combines
the sampling error of the classified sample and the error of the rectifier (two-phase variance,
Cochran ch. 12). The final taxonomy reports, for each category: the corrected share with its 95%
interval, the models-only share for comparison, the estimated number of methods in the corpus, and
precision and recall of the majority label.

**Sources.**
- Angelopoulos, Bates, Fannjiang, Jordan, Zrnic, *Prediction-Powered Inference* (Science, 2023).
- Egami, Hinck, Stewart, Wei, *Using Imperfect Surrogates for Downstream Inference: Design-based
  Supervised Learning* (NeurIPS 2023): using LLM labels directly gives substantial bias and invalid
  intervals even at 80-90% accuracy.
- *Benchmarking Debiasing Methods for LLM-based Parameter Estimates* (EMNLP 2025): finite-sample
  comparison of PPI and DSL.

**Verification.** A simulation of 400 repetitions: the true share is 30%, the models say about 41%.
The corrected share averages 30% (unbiased), and its 95% interval contains the truth more than 92% of
the time (target 95%). The models-only share stays off by more than 5 points.

---

### Problem 12 (kept): when to stop looking for new categories

**What it is.** The inductive phase draws batches of 50 methods until no new category appears. This
was already in the pipeline and is kept.

**The rule.** A name becomes a category when it reaches at least 0.5% of the labelled methods. The
loop stops after enough quiet rounds that a task with 0.5% prevalence would have appeared with 95%
probability: (1 − 0.005)^L ≤ 0.05, so L = 598 methods, 12 rounds of 50.

**Source.** De Paoli & Mathis, *Reflections on inductive thematic saturation as a potential metric
for measuring the validity of an inductive thematic analysis with LLMs* (*Quality & Quantity*); they
recommend pairing saturation with agreement measures, which steps 4 and 5 now provide.

---

### Problem 13: reproducibility and reporting

**Solution.** Every model answer is stored with its prompt hash (`labels.jsonl`,
`classify/<model>.jsonl`), so a run can be resumed and audited. When the codebook or prompt changes,
only the affected answers are redone. The report records model versions, settings, the codebook
status (the model's draft or consolidated by the researcher) and every statistic above.

**Source.** Baltes et al., *Guidelines for Empirical Studies in Software Engineering involving Large
Language Models* (Empirical Software Engineering): declare the LLM's role, report versions and
configuration, document prompts, keep interaction logs, include open models, validate against
human judgement, state limitations.

---

### Problem 14: one classifier answered "Other" by default (first run, 2026-10-19)

**What was wrong.** The first run on the real corpus (draft codebook, 34 categories, 384 new
methods) gave Krippendorff's α = 0.409, below the gate. Most of the gap came from one model:
`granite4.2:8b` answered **Other for 52% of the methods** (200 of 384), against 2.6% for qwen3.8
and 8.1% for gemma4. That includes 63 methods qwen3.8 labelled Accessor and 16 it labelled Unit
test (test methods, the easiest case of the codebook), many answered in about 0.1 s. The prompt was
not cut (8,192-token context, `models.py`). Without granite, qwen3.8 and gemma4 agree at Cohen's
κ = 0.669; each of them with granite, κ ≈ 0.30.

**Solution.** granite4.2:8b is replaced by `mistral-small3.2:24b` (Mistral AI): a fourth company, so
the three classifiers still come from three families (Problem 5), and an instruction-tuned model
with a published instruction-following score (IFEval 0.829 for Mistral Small 24B Instruct 2501,
model card; release 3.2 is described by Mistral as better at following precise instructions).
15 GB, it fits the 36 GB machine; models are run one after the other.

**How the choice avoids fitting the result.** Replacing models until α passes would be a garden of
forking paths. So: (1) the reason is a failure that does not depend on agreement (a default answer,
measured by the share of Other); (2) the replacement was chosen from published information before
it was run; (3) the new configuration is measured on a **new sample** (Problem 16); (4) the granite
run is reported here and every revision is listed in the report (`classify/revisions.json`).

**Sources.** Baltes et al. (EMSE): report every model used and why it was changed. Mistral Small 24B
Instruct model card (https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501).

---

### Problem 15: the codebook mixed kinds of categories, with one-sentence definitions

**What was wrong.** In the first run, the most frequent disagreements between qwen3.8 and gemma4
were Accessor vs Abstract method (13 methods), User interface vs Abstract method, Configuration vs
Accessor / Predicate. Some categories name the *role* of a method (Accessor, Abstract method,
Visitor pattern, Predicate), others its *domain* (User interface, Parsing, Version control): a
one-line getter of a UI class fits both, and a one-sentence definition does not say which wins.
Also, 14 of the 33 categories had entered the taxonomy early but were below 0.5% at the end of the
loop; they were still in the codebook.

**Solution.**
1. **Only categories still at ≥ 0.5% at the end go into the codebook draft** (19 in the first run).
   This is the definition the taxonomy already uses (Problem 12: the taxonomy aims at every task
   with a prevalence of at least 0.5%); the others become Other, and stay listed in `taxonomy.csv`
   for the experts. LLM-generated taxonomies tend to be over-complex (Nakashima et al., ASE 2026).
2. **Every codebook entry has a definition, "use when" and "do not use when"**, the codebook entry
   of MacQueen et al. (1998), written by the labelling model from the methods of the category and
   the list of the other categories, then consolidated by the researcher. Tested live on qwen3.8:
   the User interface entry says "do not use when the method merely retrieves or sets a single
   value (Accessor)".
3. **One decision rule**, stated in the classification prompt and in the experts' README: *classify
   a method by what its own code does, not by the class it belongs to or the application domain.*
   This is not a new construct: the open-coding prompt of step 3 already asks for "the kind of
   programming task (not the application domain)". Giving the experts the same rule matters: if
   models and experts applied different rules, the accuracy of step 5 would measure the rule, not
   the models.

**Sources.** MacQueen, McLellan, Kay, Milstein (1998), codebook structure; Hruschka et al. (2004),
iterative revision of a codebook until intercoder reliability is acceptable; Nakashima et al. (ASE
2026). **To validate:** the decision rule and the entries are the researcher's and the experts'
to confirm (`codebook.csv`, `codebook_review.csv`).

---

### Problem 16: revising the codebook on the sample that measures it

**What was wrong.** NOT_RELIABLE.md asks the researcher to read the disagreements of the classified
sample and revise the codebook. Measuring the revised codebook on the same 384 methods fits the
codebook to them: the α would be optimistic, the same double dipping as Problem 3.

**Solution.** The *instrument* is the codebook plus the list of models. When it changes, step 4 draws
a **new** sample of the same size, disjoint from step 3 and from every earlier revision (another seed
per revision, recorded in `classify/sample.json`). When nothing changed, the same sample and the
stored answers are reused. Every revision, with its α, AC1 and share of Other per model, is kept in
`classify/revisions.json` and shown in the report. The expert sample is drawn from the last
revision. A test checks that three revisions get three disjoint samples.

**Source.** The train/test separation of Problem 3 applied to each revision; Hruschka et al. (2004)
for revising in rounds; Baltes et al. (EMSE) for reporting every attempt.

---

### Problem 17: saturation of the categories is not saturation of the names

**What it is.** In the first run, 32 of the 33 categories were already in the initial 384 methods,
and the loop stopped after 13 rounds. But the candidate names (below the support threshold) kept
growing, from 45 to 100, and 13.6% of the labelled methods had a name too rare to be a category.
The categories saturated because of the 0.5% threshold, by design; the long tail did not.

**Solution.** The report now says so, with a number: the Good–Turing estimate of the probability that
the next method gets a name never seen before (names given to exactly one method / methods labelled).
It is the size of what the taxonomy leaves under Other on purpose.

**Source.** Good, I. J. (1953), *The population frequencies of species and the estimation of
population parameters*, Biometrika 40.

---

## 3. Summary table

| # | Problem | Solution | Main source |
|---|---|---|---|
| 1 | Code-completion models used to label | Instruction-tuned models, chat API; Pharo models removed | Baltes et al. (EMSE) |
| 2 | Open codes not comparable across models | Inductive codebook, then deductive classification | Shah et al. 2025; TnT-LLM (KDD 2024); Nakashima et al. (ASE 2026) |
| 3 | Codebook tested on its own data | New sample for step 4 | Cochran (1977) |
| 4 | Two models: who is wrong? | Three models, majority label, agreement level | Wisdom of the LLM Crowd (2026); Ahmed et al. (MSR 2025) |
| 5 | Correlated errors | Three companies; experts check unanimous methods too | Kim et al. (ICML 2025) |
| 6 | Kappa unsuited | Krippendorff's α, Gwet's AC1, Fleiss' κ | Krippendorff (2004); Gwet (2008) |
| 7 | Experts' time on unreliable codebook | Gate at α ≥ 0.667 | Ahmed et al.; OLAF (WSESE 2026) |
| 8 | Prompt sensitivity | Shuffled-prompt self-check | Camuffo et al. (2026) |
| 9 | Small, partly wasted expert sample | 200 methods, stratified, weighted estimators | Cochran ch. 12; Gligorić et al. (NAACL 2025) |
| 10 | Expert effort; no human baseline | CSV with the models' label filled in, corrected when wrong (anchoring documented, changes counted); taxonomy figure; two experts; consensus | Ahmed et al. (MSR 2025); Fort & Sagot (2010); Berzak et al. (2016) |
| 11 | Biased category shares | Prediction-powered inference | Angelopoulos et al. (Science 2023); Egami et al. (NeurIPS 2023) |
| 12 | When to stop | Prevalence-based saturation (kept) | De Paoli & Mathis (Q&Q) |
| 13 | Reproducibility | Every answer stored; full reporting | Baltes et al. (EMSE) |
| 14 | A classifier answering Other by default | granite4.2 replaced by mistral-small3.2, for a reason independent of α; reported | Baltes et al. (EMSE); model card |
| 15 | Overlapping categories, thin definitions | Categories ≥ 0.5% only; definition, use when, do not use when; one decision rule for models and experts | MacQueen et al. (1998); Hruschka et al. (2004) |
| 16 | Codebook revised on its own test sample | A new disjoint sample per revision; every revision reported | Problem 3; Hruschka et al. (2004) |
| 17 | Saturation of categories ≠ of names | Good–Turing estimate of unseen names in the report | Good (1953) |

---

## 4. How the work was checked

- **66 automated tests pass**, including:
  - Krippendorff's α = 0.743 and Fleiss' κ = 0.210 on the published reference examples;
  - the kappa paradox case (κ low, AC1 high);
  - the PPI simulation (unbiased, interval coverage close to 95%);
  - the stratified accuracy;
  - codebook entries (three fields), the draft leaving categories below 0.5% to Other, the
    decision rule in the prompt and in the experts' README, and three revisions getting three
    disjoint samples;
  - a full run with stand-in models and two stand-in experts, from `label` to `validate` and the
    report, checking that the step-4 sample never overlaps step 3 and that experts never see the
    models' answers.
- **Live test of the three real models** on the codebook prompt: 6/6 correct for each.

---

## 5. Limitations and open points

- **One run on the real corpus so far (2026-10-19)**, before Problems 14-17: α = 0.409, gate not
  passed. Its numbers are quoted in Problems 14, 15 and 17; the next run uses the changes.
- **The labelling model is also a classifier.** qwen3.8 named the categories and wrote the codebook,
  then classifies with it: it may agree with its own codebook more easily than the other models
  (home advantage). The experts' answers are what measure every model on equal terms;
  report qwen3.8's accuracy next to the others', not only the majority's.
- **Accessors dominate** (about 30% in step 3, 16-24% in step 4). A trivial category inflates raw
  agreement; Gwet's AC1 and the per-category precision/recall of step 5 are the numbers to read.
- **The codebook must be consolidated by the researcher** after step 3. Without `codebook.csv`, the
  model's draft is used and the report says so.
- **Majority vote, not Dawid-Skene.** Dawid-Skene would weight each model by its estimated
  reliability. It was not added because the PPI correction already removes the bias of the shares;
  it remains an option.
- **The interval of the corrected shares** uses the two-phase approximation (Cochran ch. 12). The
  simulation shows coverage close to, but slightly under, 95% in some settings (>92%).
- **The Neyman allocation uses guessed error rates** (10%, 35%, 60%). Wrong guesses make the
  estimates less precise, never biased.
- **Model ranking evidence is weak:** the choice of Qwen3.8 rests on one local comparison; the
  three-model design and the expert validation are what protect the results.
- **The expert sample size (200)** gives about ±5% on overall accuracy, less precision per category.

---

## 6. References

1. Ahmed, T., Devanbu, P., Treude, C., Pradel, M. *Can LLMs Replace Manual Annotation of Software
   Engineering Artifacts?* MSR 2025. https://arxiv.org/abs/2408.05534
2. Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., Zrnic, T. *Prediction-Powered
   Inference.* Science, 2023.
3. Baltes, S., et al. *Guidelines for Empirical Studies in Software Engineering involving Large
   Language Models.* Empirical Software Engineering. https://arxiv.org/abs/2508.15503
4. Camuffo, A., et al. *Variance-Aware LLM Annotation for Strategy Research.* arXiv 2601.02370, 2026.
   https://arxiv.org/abs/2601.02370
5. Cochran, W. G. *Sampling Techniques*, 3rd ed. Wiley, 1977 (ch. 4, 5, 12).
6. De Paoli, S., Mathis, W. S. *Reflections on inductive thematic saturation as a potential metric
   for measuring the validity of an inductive thematic analysis with LLMs.* Quality & Quantity.
   https://link.springer.com/article/10.1007/s11135-024-01950-6
7. Egami, N., Hinck, M., Stewart, B., Wei, H. *Using Imperfect Surrogates for Downstream Inference:
   Design-based Supervised Learning for Social Science Applications of LLMs.* NeurIPS 2023.
8. *Benchmarking Debiasing Methods for LLM-based Parameter Estimates.* EMNLP 2025.
   https://aclanthology.org/2025.emnlp-main.1000.pdf
9. Feinstein, A. R., Cicchetti, D. V. *High agreement but low kappa: I. The problems of two
   paradoxes.* Journal of Clinical Epidemiology, 1990.
10. Fleiss, J. L. *Measuring nominal scale agreement among many raters.* Psychological Bulletin, 1971.
11. Gligorić, K., Zrnic, T., Lee, C., Candès, E. J., Jurafsky, D. *Can Unconfident LLM Annotations Be
    Used for Confident Conclusions?* NAACL 2025. https://arxiv.org/abs/2408.15204
12. Gwet, K. L. *Computing inter-rater reliability and its variance in the presence of high
    agreement.* British Journal of Mathematical and Statistical Psychology, 2008.
13. Imran, M. M., Zaman, T. S. *OLAF: Towards Robust LLM-Based Annotation Framework in Empirical
    Software Engineering.* WSESE 2026. https://arxiv.org/abs/2512.15979
14. James, J. *Counting on Consensus: Selecting the Right Inter-annotator Agreement Metric for NLP
    Annotation and Evaluation.* LREC 2026. https://arxiv.org/abs/2603.06865
15. Kim, E. M., Garg, A., Peng, K., Garg, N. *Correlated Errors in Large Language Models.* ICML 2025.
    https://proceedings.mlr.press/v267/kim25e.html
16. Krippendorff, K. *Content Analysis: An Introduction to Its Methodology*, 2nd ed. Sage, 2004;
    *Computing Krippendorff's Alpha-Reliability*, 2011.
17. Nakashima, S., Ishimoto, Y., Kondo, M., Xiao, T., Kamei, Y. *How Well Do LLMs Generate Taxonomies
    in the SE Domain? A Multi-perspective Evaluation Framework.* ASE 2026.
    https://arxiv.org/abs/2608.01592
18. Shah, S. T. U., Hussein, M., Barcomb, A., Moshirpour, M. *From Inductive to Deductive: LLMs-Based
    Qualitative Data Analysis in Requirements Engineering.* arXiv 2504.19384, 2025.
    https://arxiv.org/abs/2504.19384
19. Wan, M., et al. *TnT-LLM: Text Mining at Scale with Large Language Models.* KDD 2024.
    https://arxiv.org/abs/2403.12173
20. *Wisdom of the LLM Crowd: A Large Scale Benchmark of Multi-Label U.S. Election-Related Harmful
    Social Media Content.* arXiv 2602.11962, 2026. https://arxiv.org/html/2602.11962
21. *Wisdom and Delusion of LLM Ensembles for Code Generation and Repair.* arXiv 2510.21513.
    https://arxiv.org/html/2510.21513
22. Good, I. J. *The population frequencies of species and the estimation of population parameters.*
    Biometrika 40(3-4), 1953.
23. Hruschka, D. J., Schwartz, D., St. John, D. C., Picone-Decaro, E., Jenkins, R. A., Carey, J. W.
    *Reliability in coding open-ended data: lessons learned from HIV behavioral research.* Field
    Methods 16(3), 307-331, 2004.
24. MacQueen, K. M., McLellan, E., Kay, K., Milstein, B. *Codebook development for team-based
    qualitative analysis.* Cultural Anthropology Methods 10(2), 31-36, 1998.
    https://doi.org/10.1177/1525822X980100020301
25. Mistral AI. *Mistral-Small-24B-Instruct-2501* model card (IFEval 0.829).
    https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
26. Fort, K., Sagot, B. *Influence of Pre-annotation on POS-tagged Corpus Development.* Fourth
    Linguistic Annotation Workshop (LAW IV), ACL 2010, 56-63. https://aclanthology.org/W10-1807/
27. Berzak, Y., Huang, Y., Barbu, A., Korhonen, A., Katz, B. *Anchoring and Agreement in Syntactic
    Annotations.* EMNLP 2016.
28. Model comparison used for the model choice: *Qwen3.8 vs Qwen3.6 vs Gemma 4: 24GB GPU Test*,
    kingy.ai, August 2026. https://kingy.ai/blog/qwen3-8-27b-vs-qwen3-6-27b-vs-gemma-4-31b/
