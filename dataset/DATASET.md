# Dataset files in this repository

The files under `dataset/` are **third-party corpora and benchmarks**, not original work in this project. They are included here for reproducibility while this repository is under active development. **Do not assume they are covered by the repository’s MIT license** (see [License](../LICENSE) and the README [License](../README.md#license) section).

For official downloads and updates, prefer the upstream sources listed below.

## Summary

| Path | Source | Upstream license | Role |
|------|--------|------------------|------|
| `train/bpcc/` | [AI4Bharat BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) / [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | Mixed — see [BPCC licenses](#bpcc-english-maithili) | Training (full English–Maithili parallel corpus) |
| `training_split/bpcc/` | Same as BPCC | Mixed — see [BPCC licenses](#bpcc-english-maithili) | Train / validation / test splits for this project |
| `test/IN22_test/gen/` | IndicTrans2 / BPCC (IN22-Gen) | **CC-BY-4.0** | General-domain benchmark |
| `test/IN22_test/conv/` | IndicTrans2 / BPCC (IN22-Conv) | **CC-BY-4.0** | Conversational benchmark |
| `test/flores_200_dev/` | [FLORES-200](https://github.com/facebookresearch/flores) | **CC-BY-SA-4.0** | Dev split for English, Hindi, Maithili |

## BPCC English-Maithili

**Upstream:** Bharat Parallel Corpus Collection (BPCC), released with IndicTrans2.

- Dataset: [Hugging Face — `ai4bharat/BPCC`](https://huggingface.co/datasets/ai4bharat/BPCC)
- Project: [GitHub — `AI4Bharat/IndicTrans2`](https://github.com/AI4Bharat/IndicTrans2)
- Paper: Gala et al. (2023), [IndicTrans2](https://arxiv.org/abs/2305.16307)

**Licenses (per AI4Bharat):** BPCC is not under a single license. The English–Maithili training mix described in the project README combines mined and human-curated portions. Relevant terms from the upstream release:

| Upstream artifact | License |
|-------------------|---------|
| Mined corpora (e.g. NLLB, Samanantar, Samanantar++) | **CC0** |
| Human seed corpora (BPCC-H-Wiki, BPCC-H-Daily) | **CC-BY-4.0** |
| Back-translation (BPCC-BT) | **CC0** |

When reusing or redistributing these files, follow the [BPCC README license table](https://huggingface.co/datasets/ai4bharat/BPCC/blob/main/README.md) and cite the IndicTrans2 paper.

**Files in this repo:**

- `train/bpcc/train.eng_Latn` — English source (one sentence per line)
- `train/bpcc/train.mai_Deva` — Maithili target (aligned line-by-line)

**Changes from upstream:**

- Exported from the IndicTrans2 / BPCC English–Maithili pair release used for this project (~67,658 parallel pairs).
- Stored as plain UTF-8 text with IndicTrans2-style language script suffixes (`eng_Latn`, `mai_Deva`).
- No additional deduplication, alignment, or toxicity filtering beyond what BPCC already provides (see root README).
- Raw NLLB English–Maithili (~4.4M pairs) was **not** included.

## BPCC train / validation / test splits

**Files in this repo:**

- `training_split/bpcc/train.{eng_Latn,mai_Deva}` — 60,892 pairs
- `training_split/bpcc/val.{eng_Latn,mai_Deva}` — 3,383 pairs
- `training_split/bpcc/test.{eng_Latn,mai_Deva}` — 3,383 pairs
- `training_split/tokenized_output.txt` — tokenizer output derived from this project’s training pipeline (not an upstream BPCC artifact)

**Changes from upstream:**

- Random split of the English–Maithili BPCC training material for local fine-tuning (proportions above); created for notebooks under `notebooks/`.
- The full corpus remains duplicated under `train/bpcc/` for convenience.
- Evaluation in the README also uses a **random sample of 1,200 pairs** (seed 42) from the 3,383-pair test split; that sample is not stored as a separate file in `dataset/`.

## IN22 (IN22-Gen and IN22-Conv)

**Upstream:** IN-22 test sets from the IndicTrans2 / BPCC release.

- **IN22-Gen:** **CC-BY-4.0**
- **IN22-Conv:** **CC-BY-4.0**

**Files in this repo:**

| Subfolder | Files |
|-----------|--------|
| `test/IN22_test/gen/` | `test.eng_Latn`, `test.mai_Deva`, `test.hin_Deva`, `metadata.tsv` |
| `test/IN22_test/conv/` | `test.eng_Latn`, `test.mai_Deva`, `test.hin_Deva`, `metadata.tsv` |

**Changes from upstream:**

- Copied benchmark parallel text in the same one-line-per-sentence format as BPCC.
- Hindi reference lines (`test.hin_Deva`) are included for comparison experiments (e.g. English → Hindi vs. English → Maithili on IN22-Gen).
- `metadata.tsv` is retained where provided by the upstream IN22 release.

## FLORES-200 (dev)

**Upstream:** FLORES-200

- Repository: [facebookresearch/flores](https://github.com/facebookresearch/flores)
- License: **CC-BY-SA-4.0** ([license text](https://github.com/facebookresearch/flores/blob/main/LICENSE_CC-BY-SA))
- Related: Costa-jussà et al. (2022), [No Language Left Behind](https://arxiv.org/abs/2207.04672)

**Files in this repo:**

- `test/flores_200_dev/dev.eng_Latn`
- `test/flores_200_dev/dev.hin_Deva`
- `test/flores_200_dev/dev.mai_Deva`

**Changes from upstream:**

- **Dev split only** (not devtest or test).
- Extracted for English, Hindi, and Maithili in plain parallel line format for this project’s evaluation notebooks.

**Note:** CC-BY-SA requires attribution and ShareAlike when you share adaptations. Treat these files as **FLORES-200 dev data under CC-BY-SA-4.0**, not as MIT-licensed project code.

## Required attribution (minimum)

When using or citing data from this folder, please cite the upstream work:

1. **BPCC / IN22:** Gala et al. (2023). *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages.* [https://arxiv.org/abs/2305.16307](https://arxiv.org/abs/2305.16307)
2. **FLORES-200 / NLLB context:** Costa-jussà et al. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* [https://arxiv.org/abs/2207.04672](https://arxiv.org/abs/2207.04672)

## This project’s code vs. data

- **MIT** applies to original code, notebooks, documentation, and project-authored artifacts (see root [LICENSE](../LICENSE)).
- **Dataset files in `dataset/`** remain under the upstream licenses above. This repository provides copies for convenience; it does not relicense them as MIT.
