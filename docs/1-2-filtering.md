# Step 1-2: filtering (`./1-2-filtering.sh`)

Keeps the repositories that are worth labelling and benchmarking: the **study corpus**, 150
repositories at most. Steps 2, 3 and 4 use nothing else.

```bash
./1-2-filtering.sh
```

Output: `results/reports/corpus_filter.md`, `results/tables/corpus_filter.csv` (log:
`results/filtering.log`). Next: [step 2](2-ai-labelling.md).

It takes about two minutes, needs no token and no model, and removes nothing from the
database.

## Why this step exists

Step 1 keeps every repository that is real Pharo code: 1,581 on the current data. That is the
right set to *find* Pharo code, and the wrong set to *study* it. Kalliamvakou et al. [1]
showed that most GitHub repositories are personal and inactive, and the mined set agrees: half
of its repositories have at most 68 commits, 2 contributors and 1 star, and a third have no
licence.

The benchmark built in step 4 asks a model to write Pharo code and to fix Pharo bugs. Its
tasks are modelled on real methods, and the statistics of step 3 say which operations those
tasks must cover. Both are only as good as the repositories behind them. A method taken from
an abandoned personal experiment, from a copy of the Pharo image, or from a project without
tests is a poor model of what a Pharo developer writes, and it cannot be redistributed when
the project has no licence.

So the question this step answers for each repository is: **is this an engineered project
whose code can serve such a benchmark?**

## What the step does

1. **Measures** every mined repository, from data step 1 already collected. Nothing is
   downloaded.
2. **Applies thirteen criteria**, one after the other. Each is a separate threshold. There is
   no combined score and no weighting.
3. **Limits the size** to `max_repositories` (150). Only when more repositories pass is a
   choice made, and it is made for diversity, never for popularity.
4. **Writes the methods step 2 will label**, taken from the study corpus: either every method
   of a number of its repositories, or a sample of methods across all of them.

## The criteria

Applied in this order. The numbers are from the current data (1,581 mined repositories).
*Removed* counts the repositories for which the criterion is the first one they fail.

| # | A repository must have | Removed | Left | Why |
|---|---|---|---|---|
| 1 | A licence that GitHub recognises | 570 | 1,011 | Tasks keep the provenance of real code and are redistributed |
| 2 | At least 2 contributors | 229 | 782 | Not a personal repository |
| 3 | At least 50 commits | 249 | 533 | Sustained development, not an upload |
| 4 | At least 180 days between creation and last push | 68 | 465 | Developed over time |
| 5 | A push in the 5 years before the mining date | 94 | 371 | Code written for a current Pharo |
| 6 | A Metacello baseline (`BaselineOf…` class) | 8 | 363 | The project says how it is loaded |
| 7 | Source in Tonel format | 39 | 324 | The source format of current Pharo |
| 8 | At least 200 methods outside tests | 92 | 232 | Enough production code |
| 9 | At most 20,000 methods | 2 | 230 | Image dumps and generated code |
| 10 | At least 20 test methods | 24 | 206 | Behaviour is specified by tests |
| 11 | At least 10% of its methods are tests | 51 | 155 | Testing as a practice, not a token test |
| 12 | At most 10% of its methods copied from another repository | 18 | 137 | Original code |
| 13 | At most 1% of its methods fail to parse | 2 | 135 | Standard Pharo syntax |

**135 repositories pass.** That is under the limit of 150, so all are kept and no further
choice is made.

The order matters only for the *Removed* column. `results/reports/corpus_filter.md` also
gives, for each criterion, how many repositories fail it whatever the others say, and
`results/tables/corpus_filter.csv` lists every mined repository with its measurements, its
decision and **every** criterion it fails.

### Why these criteria: the literature

**An engineered project, not any repository.** Munaiah et al. [2] define the target of a
mining study as the *engineered software project* and measure it along separate dimensions:
community, continuous integration, documentation, history, issues, licence and unit testing.
Criteria 1 to 4, 10 and 11 are those dimensions, measured with what is available for Pharo
repositories: community (2), history (3, 4), licence (1), unit testing (10, 11). Their paper
is also the reason a threshold is used per dimension instead of one score.

**Stars are not a criterion.** In the same paper [2], selecting repositories by a minimum
number of stars found engineered projects with high precision but missed about two thirds of
them. In Pharo the effect is stronger: half of the mined repositories have at most one star.
A star threshold would keep a handful of well-known projects and call that the ecosystem.

**Tests are what makes a task checkable.** SWE-bench [6] keeps a task only when a test fails
before the fix and passes after it, and chose popular repositories because they are better
maintained and tested. CoderEval [11] takes its tasks from functions that the original
developers tested. A benchmark for code generation and bug fixing is scored by running
tests, so the code it is modelled on must come from projects where behaviour is specified by
tests (criteria 10 and 11).

**The code must load and run today.** Multi-SWE-bench [7] requires repositories that were
actively maintained and whose latest commit builds and runs its tests in a clean environment.
The benchmark of step 4 runs in a current Pharo image (Pharo 13). Criteria 5, 6 and 7 are the
static counterpart for Pharo: recent activity, a Metacello baseline that declares how the
project is loaded, and the Tonel format that current Pharo uses. A FileTree-only project
last touched years ago often uses classes and messages that no longer exist.

**Licence.** SWE-bench [6] kept only repositories whose licence allows free use,
CodeSearchNet [10] removed projects without a licence that allows redistribution, and
RepoCoder [8] required an open-source licence. Each benchmark task records the repository,
commit and method it was derived from, so the same rule applies here (criterion 1).

**Copied code.** Lopes et al. [4] found that a large part of the code on GitHub is copied
from other projects. Allamanis [5] showed that such duplicates inflate the measured
performance of models of code, by up to 100% on some metrics. Step 1 already marks every
method whose source also exists in another repository. A repository that is mostly such
methods is a copy of something else, typically of Pharo itself (criterion 12).

**Size.** Benchmark pipelines set a minimum size so that a repository offers varied code;
SWE-Bench++ [9] requires more than 10,000 lines. Criterion 8 is the same idea at the scale
of Pharo projects. Criterion 9 removes the opposite case: a few repositories with tens of
thousands of methods are image dumps or generated bindings, and would dominate any
method-level figure.

**Above the limit: diversity, not ranking.** Nagappan, Zimmermann and Bird [3] argue that a
sample of projects should be judged by its *coverage* of the population along the dimensions
that matter, and give a greedy algorithm that adds the project increasing coverage the most.
When more than `max_repositories` pass the criteria, the corpus is chosen that way: the next
repository is always the one whose domain, size and kind are least represented so far, with
at most `max_per_owner` repositories per owner. The choice is seeded, so a rerun selects the
same repositories.

### What comes from the papers and what does not

The *criteria* come from the literature above. The *numbers* (50 commits, 200 methods, 10%)
are ours: the papers were written for Java, Python or JavaScript, where a threshold of 500
stars or 10,000 lines keeps thousands of projects and would keep almost none here. They were
set on the distribution of the mined repositories, and they are all in `config.yaml`. To see
how much a threshold matters, change it and rerun: the report shows the funnel again.

Two things are **not** measured, and are limits of this step:

- **Continuous integration and issue tracking**, two of the dimensions of [2], were not
  collected in step 1.
- **Whether the project really loads and its tests pass** in a current image. Criteria 5 to 7
  are static signals of that. Step 4 does not depend on it: its tasks are self-contained and
  each one is executed in a Pharo image before it is accepted.

## What the other steps do with the study corpus

| Step | Effect |
|---|---|
| [2: AI labelling](2-ai-labelling.md) | The methods to label come from the study corpus |
| [3: results](3-results.md) | Taxonomy, prevalence, domains and figures describe the study corpus |
| [4: benchmark](4-benchmark.md) | Seed methods are taken from the study corpus only |

The data of the other mined repositories stays in the database and in the snapshot. It is
simply not used by these steps.

## Changing the criteria

Edit `corpus_filter` in `config.yaml` and run `./1-2-filtering.sh` again.

- When the same repositories are selected, nothing else changes.
- When the corpus changes, the annotation sample and its labels no longer describe it. They
  are moved to `results/validation.before-filtering-<date>` (not deleted), a new sample is
  drawn, and steps 2, 3 and 4 must be run again.

## Settings

At the top of `1-2-filtering.sh`:

| Setting | Meaning |
|---|---|
| `MAX_REPOSITORIES` | Size limit of the study corpus. Empty = `corpus_filter.max_repositories` (150) |
| `LABEL_REPOSITORIES` | Step 2 labels every method of this many repositories. `0` or empty = a sample of methods instead |
| `SAMPLE_SIZE` | Only without `LABEL_REPOSITORIES`: methods sampled across the study corpus. Empty = `validation.annotation_sample_size` (300) |

### Two numbers: how many repositories continue, how many are labelled

- `MAX_REPOSITORIES` is how many repositories go on to the results and the benchmark.
- `LABEL_REPOSITORIES` is how many of those the models label, method by method.

They differ because labelling is the expensive part. The rule labels of step 1 exist for
every method of every repository; the AI labels are the reference they are checked against.
The repositories to label are chosen like the corpus itself when it is too large: the next
one is always the one whose domain, size and kind are least represented so far [3], with a
fixed seed. Every parsed method is taken, tests included, except methods copied from another
repository. The list is written to `results/validation/labelled_repositories.csv`.

When the methods to label change, the previous ones are moved aside with their labels (not
deleted).

In `config.yaml`, under `corpus_filter` (0 or `false` switches a criterion off):

| Key | Default | Criterion |
|---|---|---|
| `max_repositories` | 150 | Size limit |
| `require_license` | true | 1 |
| `min_contributors` | 2 | 2 |
| `min_commits` | 50 | 3 |
| `min_development_days` | 180 | 4 |
| `max_years_since_last_push` | 5 | 5 |
| `require_baseline` | true | 6 |
| `require_tonel` | true | 7 |
| `min_non_test_methods` | 200 | 8 |
| `max_methods` | 20000 | 9 |
| `min_test_methods` | 20 | 10 |
| `min_test_method_share` | 0.10 | 11 |
| `max_duplicated_method_share` | 0.10 | 12 |
| `max_parse_failure_share` | 0.01 | 13 |
| `max_per_owner` | 5 | Used only above the size limit |
| `random_seed` | 42 | Used only above the size limit |

The last-push criterion is measured from the mining date, not from today, so the same data
always gives the same corpus.

## What it writes

| Path | Contents |
|---|---|
| `results/reports/corpus_filter.md` | The funnel, the domains and the list of selected repositories |
| `results/tables/corpus_filter.csv` | Every mined repository: measurements, decision, every criterion failed |
| `results/data/processed/corpus.json` | The thresholds used, the funnel and the selected repositories |
| `results/validation/annotation_sample.csv` | The methods to label in step 2 |
| `results/validation/labelled_repositories.csv` | With `LABEL_REPOSITORIES`: the repositories labelled entirely |

## When something goes wrong

- **"No mined data yet".** Run `./1-mining.sh` first.
- **"No repository passes every criterion".** The thresholds are too strict for the data.
  Relax `corpus_filter` in `config.yaml`. Nothing was changed.
- **Steps 2, 3 or 4 say "The study corpus is not filtered yet".** Run `./1-2-filtering.sh`.

## References

1. E. Kalliamvakou, G. Gousios, K. Blincoe, L. Singer, D. M. German, D. Damian. *The Promises and Perils of Mining GitHub.* MSR 2014; extended in Empirical Software Engineering 21, 2016
2. N. Munaiah, S. Kroh, C. Cabrey, M. Nagappan. *Curating GitHub for Engineered Software Projects.* Empirical Software Engineering 22(6), 2017
3. M. Nagappan, T. Zimmermann, C. Bird. *Diversity in Software Engineering Research.* ESEC/FSE 2013
4. C. V. Lopes et al. *DéjàVu: A Map of Code Duplicates on GitHub.* OOPSLA 2017
5. M. Allamanis. *The Adverse Effects of Code Duplication in Machine Learning Models of Code.* Onward! 2019. arXiv:1812.06469
6. C. E. Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. arXiv:2310.06770
7. D. Zan et al. *Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving.* 2025. arXiv:2504.02605
8. F. Zhang et al. *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation.* EMNLP 2023. arXiv:2303.12570
9. *SWE-Bench++: A Framework for the Scalable Generation of Software Engineering Benchmarks from Open-Source Repositories.* 2025. arXiv:2512.17419
10. H. Husain et al. *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search.* 2019. arXiv:1909.09436
11. H. Yu et al. *CoderEval: A Benchmark of Pragmatic Code Generation with Generative Pre-trained Models.* ICSE 2024

The claims taken from references 1 to 11 were checked against the published pages or
abstracts while writing this file, with two limits. For [2], the list of dimensions and the
result on stars were checked, but not the paper's own threshold values. For [7], [9] and
[10], the criteria were read from search summaries of the papers, not from the full texts.
Author lists, venues and years are cited from memory: check them before using them in a
paper.
