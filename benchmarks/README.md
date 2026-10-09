# Benchmark task metadata

`pharo-miner benchmark-coverage --input benchmarks/<file>` imports existing
benchmark tasks and compares them with the validated taxonomy.

Accepted formats: `.json` (list or `{"tasks": [...]}`), `.jsonl`, `.yaml`, `.csv`
(list fields separated by `;`).

```json
{
  "benchmark": "example_benchmark",
  "task_id": "task_001",
  "description": "...",
  "operations": ["collection_filtering", "string_conversion_normalization"],
  "structure": ["single_method"],
  "required_capabilities": ["collections", "blocks"],
  "transfer_status": "not_evaluated",
  "difficulty": "easy",
  "preserves_capability": true,
  "origin": "HumanEval/12"
}
```

- `operations` must be **taxonomy level-3 ids assigned by a person** who read the
  task. Unknown ids are reported and ignored. Descriptions are never keyword-matched.
- `transfer_status`: `not_evaluated`, `portable`, `needs_adaptation`,
  `transferred_validated`, `native_pharo`, `not_transferable`.
- `preserves_capability`: whether the (transferred) task still exercises the
  intended Pharo capability (`true`/`false`; empty = unknown).

`template.json` is an empty template: there is no benchmark data in this
repository, and none is fabricated.
