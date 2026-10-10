# Step 4: benchmark generation (`./4-benchmark.sh`)

How it works and why it can be trusted. This file explains how programming tasks for a
Pharo benchmark are written automatically, and how each one is proven correct without anyone
reading it.

## The idea in one paragraph

Steps 1 to 3 tell us which programming operations real Pharo code uses. Step 4 writes
benchmark tasks for those operations. A language model writes each task, but **no model
decides whether a task is kept**. Every task is installed in a clean Pharo image and run. It
is kept only if ten checks pass, and each check is a fact observed by executing code. A task
that fails is sent back to the model with the image's error message a few times, then
discarded. The discarded tasks and the reason for each are kept and reported.

## What a task is

| Part | What it is | Who sees it |
|---|---|---|
| Statement | Text that says what to implement | The model being evaluated |
| Class and context methods | A small class, with `initialize`, accessors and helpers | The model being evaluated |
| Target methods | One or two methods to implement; only their signature is shown | The model being evaluated |
| Reference solution | A correct implementation of the target methods | Nobody; it validates the task |
| Tests | An SUnit test class that decides whether an implementation is correct | Nobody; it scores the answer |
| Provenance | The real method the task was derived from (repository, commit, file, line) | The reader of the benchmark |

Tasks are self-contained: they use only classes of a standard Pharo image, so the benchmark
installs anywhere without dependencies.

## The pipeline

```
study corpus ──> seed methods ──> model writes solution + tests ──> Pharo image checks them
                                        ^                                   │
                                        └──── error report / surviving ─────┘
                                              mutants (a few rounds)
        ──> model writes the statement ──> a second model solves it from the statement alone
        ──> all accepted tasks are installed together through Metacello ──> benchmark
```

1. **Seeds.** For each operation, real methods are taken from the study corpus (the
   repositories kept by [step 1-2](1-2-filtering.md): licensed, tested, original): methods
   with strong rule evidence of the operation, not tests, not duplicates, 5 to 40 lines, at
   most one per repository. The choice is a fixed pseudo-random order, so a rerun selects the
   same methods. The operations are those the coverage analysis of step 3 recommends (or all
   level-3 operations when that analysis was not run).
2. **Solution and tests first.** The model sees the seed and the definition of the operation,
   and writes a *new* small problem that exercises the same operation: a class, a reference
   solution and tests. It is told not to copy the seed, and a check enforces it.
3. **Validation in the image**, described below. On failure the model gets the exact error
   (for example "`keysWhereValue:` is not implemented anywhere in the image") and tries again.
4. **Tests against surviving mutants.** When some wrong variants of the solution still pass
   the tests, they are shown to the model, which must add tests that tell them apart.
5. **Statement last.** The statement is written only after the code is validated, from the
   validated code. This order is deliberate: the statement describes behaviour that is
   already known to be executable and tested.
6. **Independent solution.** A different model receives only what a benchmark user receives
   (statement, class, context methods, signatures) and must produce a solution that passes
   all tests. It never sees the reference or the tests.
7. **Joint installation.** All accepted tasks are exported as Tonel packages with a Metacello
   baseline, loaded together in one clean image, and all tests are run again.

## The ten checks

A task is accepted only if all ten pass. They run in this order, and the first one that fails
is the recorded reason, so the counts in the report read as a filtering funnel.

| # | Check | What is observed | What it protects against |
|---|---|---|---|
| 1 | `well_formed` | Required parts are present; at least 4 tests; at most 2 target methods | Empty or oversized tasks |
| 2 | `installs` | Every method parses and compiles in the clean image; no class name clash; no undeclared name; **every selector sent is implemented somewhere in the image** | Syntax errors and invented API, the most common error of models writing Pharo |
| 3 | `hermetic` | The code references no file, network, clock, UI, OS or unseeded random API | Tests that depend on the machine or on chance |
| 4 | `reference_passes` | All tests pass on the reference solution, 3 times, including once in reverse order, each test within 10 s | Wrong expected values, order-dependent and flaky tests, infinite loops |
| 5 | `fail_to_pass` | For each target method: when it is replaced by a stub, at least one test fails | Tests that do not exercise the code they are meant to test |
| 6 | `mutation` | MuTalk mutates the reference solution; the tests must detect at least 80% of the mutants, with at least 4 mutants | Tests that run the code but do not check its behaviour |
| 7 | `operation_preserved` | The project's own parser and rule classifier, run on the exported task, find the intended operation in the reference solution | Tasks that drifted away from the operation they are meant to measure |
| 8 | `solvable` | A second model, given only the statement, writes a solution that passes all tests; the statement contains no code of the reference | Statements that are ambiguous, incomplete or inconsistent with the tests |
| 9 | `unique` | Class names and solutions are not repeated; the solution is not a copy of the seed (token similarity at most 0.6) | Duplicates, and third-party code copied into the benchmark |
| 10 | `installs_together` | The whole benchmark loads through its Metacello baseline in one clean image and each task's tests still pass | Tasks that only work alone, or a benchmark that cannot be installed |

The thresholds are in `config.yaml` under `benchmark_generation`.

### Why these checks: the literature

**Models propose, execution decides.** Recent benchmark-construction pipelines generate with
a model and filter by execution in a sandbox. AutoCodeBench [1] generates solutions first,
obtains test outputs by running them in a sandbox, and writes the problem statement last
("reverse-order generation"), followed by several filtering steps. CodeBenchGen [2] turns
real GitHub code into executable tasks with model-generated tests and an iterative
execute-and-repair loop. Steps 2, 3 and 5 of this pipeline follow that design.

**A passing reference is not enough.** EvalPlus [3] showed that HumanEval's tests were too
weak: after adding tests, measured pass rates dropped by up to 19 to 29 points and model
rankings changed. UTBoost [4] found the same in SWE-bench Verified, a benchmark that had been
reviewed by people: 26 tasks with insufficient tests and hundreds of wrong patches counted as
correct. The conclusion is that test adequacy has to be measured, not assumed, which is what
checks 5 and 6 do.

**Fail-to-pass.** SWE-bench [5] validates each task by requiring tests that fail before the
fix and pass after it. Check 5 is the same criterion applied per target method.

**Mutation testing.** A mutant is the solution with one small fault injected (a `>=` turned
into `>`, a `select:` into `reject:`, a removed `^`). A test suite that still passes has
missed that fault. The technique comes from DeMillo, Lipton and Sayward [6] and is surveyed by
Jia and Harman [7]. The reason it is a valid measure here is the result of Just et al. [8]:
across 357 real faults, a suite's ability to detect mutants correlates with its ability to
detect real faults, independently of code coverage. MuTalk [9] is the mutation framework of
Pharo; it is used unchanged (version 3.0.8, its default mutation operators, every test run
against every mutant).

**Tests written against surviving mutants.** Feeding surviving mutants back to a model to
obtain tests that detect them is the approach of Meta's ACH system [10], and EvalPlus [3] uses
mutant detection to select tests. Step 4 of the pipeline does this, and each added test must
itself pass on the reference in the image before it counts.

**Flaky tests.** Luo et al. [11] identify test-order dependence, concurrency and time as main
causes of flaky tests. Check 4 reruns in reverse order, and check 3 excludes the clock,
randomness and the outside world.

**Solvable from the statement.** CodeBenchGen [2] validated solvability with a human study.
Here the same question is answered automatically by a second model that sees only the
statement. Three artefacts are thereby tied together by execution: tests agree with the
reference (check 4), the statement agrees with the tests (check 8), and the tests are strong
(checks 5 and 6).

**Grounded in real code.** CoderEval [12] and ClassEval [13] argue that benchmarks should
reflect how code is written in real projects rather than isolated puzzles. Here every task
starts from a real method, and check 7 verifies with the same classifier used for the corpus
statistics that the final task still contains the operation. This is what connects a
benchmark score to the prevalence figures of step 3.

## The image

"Validated in a specific image" means:

- Pharo 13 (64-bit), downloaded once from `get.pharo.org`, with MuTalk v3.0.8 loaded through
  Metacello. It is stored in `results/benchmark/pharo/`.
- The exact build string of the image, the sha256 of the image file, the MuTalk version and
  a hash of the harness scripts are written to `manifest.json` and to the report.
- **Every task runs in its own throw-away copy of that image**, which is deleted afterwards
  and never saved. No task can see or disturb another, and tasks are validated in parallel.
- A task that crashes the image or runs longer than 10 minutes is rejected.

The code that runs inside the image is in `pharo/BenchmarkHarness/`: `harness.st` (one task)
and `install.st` (the whole benchmark through its baseline).

## What "correct without manual validation" means, and what it does not

What is established for every accepted task, by execution:

- It installs in the pinned image, alone and together with all other tasks.
- Its tests pass on the reference solution, repeatedly and in any order.
- Its tests fail when the solution is missing, and detect at least 80% of injected faults.
- Its reference solution contains the operation it claims to measure.
- Its statement was sufficient for an independent model to solve it.

What is **not** established, and should be stated as a threat to validity:

- **The statement may be poorly worded** even if it is sufficient. No check reads prose.
- **Tests may be stricter than the statement.** A valid alternative solution could fail
  because a test expects one particular output format. The independent solution reduces this
  risk (one different implementation did pass) but does not remove it.
- **Surviving mutants.** Up to 20% of mutants may survive. Some are equivalent to the
  original (no test can detect them); others are real gaps. They are listed per task.
- **The solvability check favours easier tasks**, because a task that the solver model cannot
  solve is discarded even when it is correct. The share rejected by this check is reported.
  Use a strong solver, and one from a different family than the author.
- **Some operations cannot be tested hermetically** (files, HTTP, databases, OS processes,
  UI). Their candidates are rejected by check 3. The benchmark therefore covers fewer
  operations than the taxonomy, and the report shows which.
- **Check 7 inherits the limits of the rule classifier.** It confirms that the operation's
  characteristic messages are present, not that the task is a good exercise of it.

For a paper, a small human audit of a random sample of accepted tasks is still worth doing.
Its role is to *measure* the residual error of the automatic pipeline (as UTBoost did for
SWE-bench), not to make the pipeline work.

## How to run it

```bash
./4-benchmark.sh
```

Settings are at the top of the file:

| Setting | Meaning |
|---|---|
| `AUTHOR` | The model that writes the tasks (a name from `models.yaml`). Use the strongest one available |
| `SOLVERS` | Models that must solve each task from its statement. Use a different model than `AUTHOR`; `none` skips check 8 and the report says so |
| `OPERATIONS` | `auto`, or a list of level-3 operation ids |
| `TASKS_PER_OPERATION` | Seed methods, and so candidate tasks, per operation (default 3) |
| `LIMIT` | Only the first N seeds: a cheap trial |

It can be interrupted and rerun: model answers and decided tasks are kept.

To check an existing benchmark again, with no model involved:

```bash
.venv/bin/pharo-miner benchmark-validate            # results/benchmark/tasks.jsonl
.venv/bin/pharo-miner benchmark-validate --input my_tasks.json --strict
```

This reruns checks 1 to 10 in the image (check 8 reruns the stored independent solution) and
writes everything to `results/benchmark/verification/`. Hand-written tasks can be admitted
through the same checks this way.

## Outputs

| Path | Contents |
|---|---|
| `results/benchmark/tasks.jsonl` | Accepted tasks: statement, prompt, class, reference solution, tests, independent solution, seed provenance, the result of every check, mutation counts and surviving mutants |
| `results/benchmark/rejected.jsonl` | Discarded candidates with the check each one failed and the history of attempts |
| `results/benchmark/PharoTaskBench/` | The benchmark as Tonel packages (one per task) with `BaselineOfPharoTaskBench` |
| `results/benchmark/manifest.json` | Image identity, models, thresholds, filtering funnel, mutation scores |
| `results/benchmark/benchmark_tasks.json` | The tasks in the format `pharo-miner benchmark-coverage --input` reads |
| `results/benchmark/llm_cache/` | Every model answer |
| `results/benchmark/work/` | Per task and attempt: the job sent to the image and its raw result |
| `results/reports/benchmark_generation.md` | The report: funnel, test strength, accepted and rejected tasks |

Installing the benchmark in any Pharo 13 image:

```smalltalk
Metacello new
	baseline: 'PharoTaskBench';
	repository: 'tonel:///path/to/results/benchmark/PharoTaskBench/src';
	load.
```

## How the validator itself was tested

A validator that accepts everything would also "work", so it is tested against tasks built
to be wrong (`tests/fixtures/benchmark_tasks.json`, `tests/test_benchgen.py`). Twelve
hand-written tasks were run in the real Pharo 13 image; each was rejected by the check it was
written for:

| Fixture | Defect | Rejected by |
|---|---|---|
| `good-01` | none | accepted, 17 of 17 mutants detected |
| `syntax-error-01` | unbalanced bracket | `installs` |
| `unknown-selector-01` | sends `keysWhereValue:`, which does not exist | `installs` |
| `clock-01` | uses `DateAndTime now` | `hermetic` |
| `wrong-oracle-01` | one test expects a wrong value | `reference_passes` |
| `flaky-01` | a test that passes only the first time it runs | `reference_passes` (not deterministic) |
| `loop-01` | infinite loop in the solution | `reference_passes` (timeout) |
| `no-fail-to-pass-01` | tests never call the target method | `fail_to_pass` |
| `weak-tests-01` | tests only check `isEmpty` / `notEmpty` | `mutation` (12 of 17 detected) |
| `wrong-operation-01` | labelled as JSON serialization, does filtering | `operation_preserved` |
| `bad-solution-01` | stored independent solution uses `>` instead of `>=` | `solvable` |
| `zz-duplicate-01` | same class name as `good-01` | `unique` |

A second test drives the whole generation loop with scripted model answers: a first answer
with a wrong expected value (repaired), then weak tests (strengthened against the surviving
mutants), then a statement, then one wrong and one right independent solution. The task ends
accepted with all ten checks passed and a mutation score of 1.0.

These tests run with `.venv/bin/python -m unittest discover -s tests`; the ones that need the
image are skipped until it has been built.

**What has not been done yet:** no benchmark has been generated with a strong model. A trial
with two local 7B models on three operations produced three candidates, and the image
rejected all three (invented selectors, names copied from the seed, tests that did not
exercise the target). That is the expected result for models of that size, and it shows the
checks hold under real model output; it says nothing yet about the acceptance rate of a
strong model.

## References

1. Tencent Hunyuan Team. *AutoCodeBench: Large Language Models are Automatic Code Benchmark Generators.* ICLR 2026. arXiv:2508.09101
2. Y. Xie et al. *CodeBenchGen: Creating Scalable Execution-based Code Generation Benchmarks.* 2024. arXiv:2404.00566
3. J. Liu et al. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS 2023. arXiv:2305.01210
4. B. Yu et al. *UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench.* ACL 2025. arXiv:2506.09289
5. C. E. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. arXiv:2310.06770
6. R. A. DeMillo, R. J. Lipton, F. G. Sayward. *Hints on Test Data Selection: Help for the Practicing Programmer.* IEEE Computer 11(4), 1978
7. Y. Jia, M. Harman. *An Analysis and Survey of the Development of Mutation Testing.* IEEE TSE 37(5), 2011
8. R. Just, D. Jalali, L. Inozemtseva, M. D. Ernst, R. Holmes, G. Fraser. *Are Mutants a Valid Substitute for Real Faults in Software Testing?* FSE 2014
9. MuTalk, <https://github.com/pharo-contributions/mutalk>; M. Abdi, S. Demeyer. *Steps towards Zero-touch Mutation Testing in Pharo.* CEUR-WS Vol. 3245, 2022
10. C. Foster et al. *Mutation-Guided LLM-based Test Generation at Meta.* FSE Companion 2025. arXiv:2501.12862
11. Q. Luo, F. Hariri, L. Eloussi, D. Marinov. *An Empirical Analysis of Flaky Tests.* FSE 2014
12. H. Yu et al. *CoderEval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models.* ICSE 2024
13. X. Du et al. *ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation.* ICSE 2024

References 1 to 4, 8, 9 and 10 were checked against their published pages while writing this
file. The others are cited from memory: check the bibliographic details before using them in
a paper.
