# Snapshots on Hugging Face (`./snapshot.sh`)

A snapshot is a copy of all results, saved in a Hugging Face dataset. It exists so that
nobody has to mine GitHub twice: mining takes hours, downloading a snapshot takes minutes.
It is also how results travel between a laptop and the cluster.

The dataset is set in `config.yaml` (`snapshot.repo`, default
[`pharo-llm/pharo-labels`](https://huggingface.co/datasets/pharo-llm/pharo-labels)).

## Commands

```bash
./snapshot.sh push              # upload the current results as a new version
./snapshot.sh push --dry-run    # show what would be uploaded, upload nothing
./snapshot.sh pull              # download the latest snapshot (only when there are no local results)
./snapshot.sh pull --force      # replace local results (the old ones are kept, see below)
./snapshot.sh pull --revision snapshot-20261009-153000    # a specific older version
./snapshot.sh info              # what is local, what is online, the list of versions
```

Every version is tagged `snapshot-YYYYMMDD-HHMMSS`.

## When the step scripts use it

- `./1-mining.sh` downloads the latest snapshot when there are no local results, instead of
  mining.
- Steps 1, 2 and 3 each upload a new version when they finish, if `PUSH_SNAPSHOT="yes"` at
  the top of the script.
- Step 4 does not upload anything.

## What a snapshot holds

The evidence database, the parsed methods, the discovery data, the raw GitHub responses, the
annotation sample, human and AI annotations (every model answer), tables, reports and logs.
`snapshot/manifest.json` has a sha256 for every file.

Not included:

- **Cloned repositories** (third-party code), unless `snapshot.include_clones: true`. They can
  be downloaded again from the recorded commits.
- **E-mail addresses** in raw GitHub responses. They are replaced by `<redacted>` before the
  upload.
- **The generated benchmark** (`results/benchmark/`). Only its report in `results/reports/`
  is included.

## Safety

- A pull verifies every checksum before installing anything.
- A pull never deletes local results. With `--force`, the old `results/` is renamed to
  `results.before-pull-<date>`.
- Uploading needs a Hugging Face token with write access (`HF_TOKEN`, asked once, kept in
  `.env`). Downloading a public dataset needs none.

## If an upload fails

The results stay on the computer; nothing is lost. Run `./snapshot.sh push` again. Until it
succeeds, other machines still get the previous version.
