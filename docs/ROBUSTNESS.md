# Robustness — Velo Base 1.0

This page reports **first-party** presentation and alien-key stress for Velo only. Competitor systems were **not** re-run on the ultra-fresh suite for these public tables.

Artifact: [`../artifacts/robustness_ultra_fresh.json`](../artifacts/robustness_ultra_fresh.json).

## Label honesty

Gold labels are **only** official `LocalLLaMA/typed-decisions` teacher soft/hard labels. Transforms mutate presentation (casing, wrappers, synonyms, dates, noise). No hand-invented gold. Agreement with any other model is never used as gold.

## Packaging (no retrain)

Sealed heads stayed fixed. Inference packaging added:

- Sorted choice keys at feature/parse time
- `prepare_state` flatten / wrapper unwrap (`dossier`, `satchel`, `transit_bay`, …)
- Broader aliases + fuzzy canonicalize
- Opaque key remapping heuristics
- Date / unicode normalization helpers

Train already used snake_case + alphabetically sorted choice keys, so weights did not need refit for those fixes.

## Gates (self-imposed)

| Gate | Result |
|---|:---:|
| Presentation all ≥ **0.60** (must) | **PASS** |
| Presentation all ≥ **0.72** (stretch) | **FAIL** |
| Sealed clean accuracy ≥ **0.752** | **PASS** |

Overall must-gates: **PASS / GO**. Stretch ≥0.72 is informational.

## Baseline (sealed clean)

| n | accuracy | soft_acc | ECE |
|---:|---:|---:|---:|
| 2000 | 0.7530 | 0.5574 | 0.1195 |

(Headline sealed calibrated accuracy for the release is **0.7535**; suite baseline row above is the ultra-fresh harness readout.)

## Presentation buckets (Velo)

| bucket | n | accuracy | ≥60 | ≥72 |
|---|---:|---:|:---:|:---:|
| baseline | 2000 | 0.7530 | YES | YES |
| option_scramble | 2000 | 0.7530 | YES | YES |
| known_synonym_camel | 2000 | 0.7530 | YES | YES |
| casing_all_caps | 2000 | 0.7390 | YES | YES |
| casing_title | 2000 | 0.6760 | YES | **NO** |
| nested_wrappers | 2000 | 0.7530 | YES | YES |
| numeric_string_formats | 2000 | 0.7380 | YES | YES |
| research_sci_numeric | 2000 | 0.7430 | YES | YES |
| date_time_formats | 2000 | 0.7415 | YES | YES |
| date_unicode_pad | 2000 | 0.7425 | YES | YES |
| extra_finance_noise | 2000 | 0.7395 | YES | YES |
| marketing_cs_padding | 500 | 0.6460 | YES | **NO** |
| coding_agent_noise | 500 | 0.7100 | YES | **NO** |
| missing_peripheral | 2000 | 0.7495 | YES | YES |
| deep_synonym_nest | 2000 | 0.7530 | YES | YES |
| combo_hard | 2000 | 0.7180 | YES | **NO** |
| combo_ultra | 2000 | 0.7125 | YES | **NO** |
| finance_flavored | 1000 | 0.7670 | YES | YES |

Presentation macro-accuracy (excl. baseline & alien): **~0.731**.

Stretch fail modes to call out: Title-Case, CS shoutcase padding, coding noise, hard/ultra combos — all still clear the 60% must-bar.

## Alien keys (outside synonym maps)

Reported separately; not mixed into the ≥60/≥72 presentation claim.

| bucket | n | accuracy |
|---|---:|---:|
| alien_keys | 2000 | 0.7510 |
| alien_finance_vocab | 1000 | 0.7890 |

**Alien macro-accuracy: ~0.770.**

These renames are intentionally outside `FIELD_SYNONYMS` / broader aliases used at packaging time. Drops measure unmapped-key brittleness, not label noise.

## Limits

- Presentation transforms do not re-query a teacher on mutated text.
- Deep nesting / alien renames change feature paths — drops are expected.
- Numeric currency/comma strings can stress float heuristics; count predict failures separately when they occur.
- Suite seed `20260929`; wrappers `dossier` / `satchel` / `transit_bay` were fresh relative to earlier internal suites.
- **Laya and Jev were not re-run on this suite** for the public release tables.
