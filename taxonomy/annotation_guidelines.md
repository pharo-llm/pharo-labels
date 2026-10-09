# Annotation guidelines — PharoTaskMiner operation taxonomy

These guidelines are for researchers reviewing methods sampled from real Pharo
repositories. The goal is to check whether the **draft taxonomy** describes the
programming operations that methods actually perform. It is *not* to judge code
quality, developer productivity or difficulty.

## Files

| File | Who edits it | Purpose |
|---|---|---|
| `annotation_sample.csv` | nobody | Methods to annotate (source, provenance). Automatic labels are hidden (blind annotation). |
| `annotation_assignments.csv` | nobody | Which annotator reviews which item. Items marked `overlap=1` are reviewed by every annotator to measure agreement. |
| `reviewer_decisions.csv` | annotators | One row per (sample_id, annotator). Fill in your answers. |
| `adjudicated_labels.csv` | adjudicator | Created by `import-annotations`; resolve rows with `status=needs_adjudication`. |
| `developer_feedback.csv` | Pharo developers | Taxonomy-level feedback once the draft is stable. |

Work independently. Do not look at another annotator's answers or at
`annotation_key.csv` until adjudication.

## For every method

1. Read the whole method, its class and selector. Use the repository link if
   the context is unclear.
2. In `operations`, list **every** level-3 category id (from the table at the
   end of this file) that the method *performs*, separated by `;`.
   - Label what the method does, not what it merely mentions. A `select:` used
     to filter a collection is `collection_filtering`; a method whose only
     statement returns an instance variable is `state_access_modification`.
   - Label control structures (`boolean_conditionals`, `conditional_loops`, …)
     only when they are a meaningful part of the method's behaviour.
   - If no category fits, write `unknown` and describe the operation in
     `new_category` (short snake_case name) and `notes`.
   - If the method is too small or context-dependent to judge, write
     `insufficient_context`.
3. Answer the review questions (use `1` for yes, leave empty for no):
   - `too_broad` — an assigned category lumps together clearly different operations.
   - `too_narrow` — a category is so specific it would only fit very few methods.
   - `multiple_operations` — the method combines several distinct operations.
   - `evidence_sufficient` — `yes`/`no`/`partly`: is the source alone sufficient to decide?
   - `missing_operations` — operations you see that the taxonomy does not name.
4. Keep `notes` short and concrete (e.g. "builds a Roassal shape per node, then lays out").

## Decision rules

- Multi-label is expected. Do not force a single label.
- Dynamic dispatch: do not assume receiver types that the code does not show.
  `x add: y` is collection modification only when `x` is evidently a collection.
- Framework-specific operations (e.g. `shapes_canvas_rendering`) apply only when
  the framework's API is actually used.
- Test methods: label what the test exercises *and* `unit_test_assertions`.
- When unsure between two categories, choose both and say so in `notes`.

## Adjudication

After `pharo-miner import-annotations`:

- `disagreements.csv` lists items where annotators disagree.
- In `adjudicated_labels.csv`, fill `operations` for rows with
  `status=needs_adjudication`, set `adjudicator`, and optionally `notes`.
  Rows with `agreed`/`single_annotation` are pre-filled; correct them if needed.
- Re-run `pharo-miner import-annotations`. Only adjudicated labels count as
  validated; automatic labels are never treated as validated.

## Developer feedback

Respondents are identified by a pseudonym or role only. For each category, rate
1–5 whether it is recognisable and important in their work, whether it deserves
benchmark coverage (yes/no), whether the examples genuinely represent Pharo
programming, and list missing capabilities.
