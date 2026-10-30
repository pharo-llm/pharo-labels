# Estimated API bill for the full pipeline run

This estimates what the full run (job 337451, Grid'5000 vianden, 26–27 Oct 2026, PROFILE=full) would have cost on paid APIs instead of the local Ollama models.
- **Token counts:** the exact tokens sent and received, from each phase's metrics (`results/*/metrics`). Retries are included; embeddings (nomic, under $1) are not.
- **Assumption:** one model does all the AI work in every phase.
- **Actual cost:** $0, because Grid'5000 is free for research.

## Cost per phase (USD)

| Phase | Tokens (in / out) | GPT-6 Astra | GPT-6.1 Sol | GPT-6.1 Terra | GPT-6 Luna | Claude Fable 5.1 | Claude Opus 5.5 | Claude Sonnet 5.5 | Claude Haiku 5.5 |
|---|---|---|---|---|---|---|---|---|---|
| 1–4 Repos, literature, corpus, sampling | no AI | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 Discovery | 4.9M / 1.6M | 130 | 26 | 13 | 1.30 | 130 | 52 | 26 | 1.30 |
| 6 Taxonomy | 0.55M / 0.40M | 25 | 5 | 2.50 | 0.25 | 25 | 10 | 5 | 0.25 |
| 7 Annotation | 90.8M / 0.46M | 932 | 186 | 93 | 9.30 | 932 | 373 | 186 | 9.30 |
| 8 Validation | no AI | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 Profile | 56.1M / 0.23M | 573 | 115 | 57 | 5.70 | 573 | 229 | 115 | 5.70 |
| 10 Coverage | no AI | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 Benchmark | 0.61M / 0.37M | 24 | 4.90 | 2.40 | 0.24 | 24 | 9.70 | 4.90 | 0.24 |
| 12 Verification | 0.04M / 0.03M | 2.10 | 0.42 | 0.21 | 0.02 | 2.10 | 0.85 | 0.42 | 0.02 |
| 13 Evaluation | 0.03M / 0.004M | 0.43 | 0.09 | 0.04 | 0.00 | 0.43 | 0.17 | 0.09 | 0.00 |
| **Total** | **153M / 3.1M** | **$1,687** | **$337** | **$169** | **$17** | **$1,687** | **$675** | **$337** | **$17** |

## Prices used (USD per 1M tokens, input / output)

| OpenAI | Price | Claude | Price |
|---|---|---|---|
| GPT-6 Astra | $10 / $50 | Fable 5.1 | $10 / $50 |
| GPT-6.1 Sol | $2 / $10 | Opus 5.5 | $4 / $20 |
| GPT-6.1 Terra | $1 / $5 | Sonnet 5.5 | $2 / $10 |
| GPT-6 Luna | $0.10 / $0.50 | Haiku 5.5 | $0.10 / $0.50 |

**Sources:**
- OpenAI: the official pricing page, https://developers.openai.com/api/docs/pricing (standard tier, short context), checked 2026-10-27.
- Claude: Anthropic's list prices as of 2026-10-06.

## What this means

- **Phases 7 and 9 are about 95% of the bill.** Every call in those phases sends the whole taxonomy, and they make 4,348 calls in total.
- **Prompt caching would lower these numbers a lot.** The taxonomy part of the prompt is the same in every call, and both companies charge about 10% for repeated text. With caching, Opus would be roughly $200 instead of $675, and the other models would drop by about the same share.
- **The real bill could be higher.** The paid models "think" before answering, which adds output tokens that the local models did not produce.
- **A cheaper mixed setup is possible.** For example, a cheap model for phases 7 and 9 and a strong model for phases 5, 6 and 11 would come to about $20–100 in total.
