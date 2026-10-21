# Midterm questions E1–E4

Answers to the four Midterm questions of the brief. Part 1 uses three seeds (42/43/44); Part 2 uses the original
single Colab run (seed 42, `python scripts/midterm_questions.py`, 758 clean test clips). No model is re-run and nothing was tuned on the test set: checkpoints were
selected on validation macro-F1.

> **Update (three seeds).** Part 1 below is the current answer: the models were re-run locally with seeds 42/43/44
> (RTX 2060 SUPER) and every number is mean ± std over the three runs
> (`results/tables/midterm_ms_*.csv`, produced by `scripts/midterm_multiseed.py`; baselines in
> `results/tables/midterm_baselines.csv`). Part 2 is the original Midterm answer from the single Colab run
> (seed 42, kept in `results/colab_seed42/`). It is kept as a record only: some of its conclusions do **not**
> survive the repeat (the model ranking in E4, the CRNN-worse-on-long-clips result in E3).

Many classes have only 8–17 test clips, so one clip moves a per-class recall by 6–12 points, in both parts.

# Part 1. Current answers (three seeds, mean ± std)

Clean test set, 758 clips. Checkpoints were selected on validation macro-F1 and nothing was tuned on the test set.

| model | test macro-F1 | test accuracy |
|---|---:|---:|
| `small_cnn` | 0.876 ± 0.015 | 0.931 ± 0.012 |
| `mobilenetv3s` | 0.901 ± 0.027 | 0.949 ± 0.016 |
| `crnn` | 0.875 ± 0.011 | 0.931 ± 0.005 |

The seed-to-seed spread (up to 0.027) is as large as the gap between the models, so the three are not separated on
clean data. The Colab run had `crnn` at 0.815, mainly because it stopped early (best epoch 12); locally it trains
32–46 epochs to its best checkpoint.

## E1 (3 seeds). Which classes are recognised best and worst?

Per-class F1 averaged over the three models and three seeds ([`midterm_ms_e1_per_class.csv`](../results/tables/midterm_ms_e1_per_class.csv)).

| | class | n test | mean F1 | std |
|---|---|---:|---:|---:|
| best | Voices | 285 | 0.995 | 0.002 |
| | Chicken coop | 9 | 0.988 | 0.023 |
| | Wind turbine | 15 | 0.975 | 0.021 |
| | Music | 150 | 0.974 | 0.011 |
| | Cicadas and crickets | 11 | 0.974 | 0.025 |
| worst | Jet aircrafts | 15 | 0.706 | 0.068 |
| | Train | 11 | 0.730 | 0.092 |
| | Vehicle idling | 17 | 0.744 | 0.059 |
| | Vacuum cleaner fan and hairdryer | 15 | 0.756 | 0.076 |
| | Propeller aircrafts | 15 | 0.757 | 0.077 |

The best and worst ends are the same kind of class as in the single run: easy classes are Voices, Music and Wind
turbine; hard classes are stationary engine / machine sounds. The caveat about curation shortcuts (Voices and Music cut
off at 11.025 kHz, fixed clip lengths for Music and Wind turbine, see Part 2 E1) still applies. The exact order inside
the worst group changes with the seed (std 0.06–0.09), so only the group should be quoted.

## E2 (3 seeds). Which classes are confused most?

Errors per unordered class pair, averaged per run, summed over the three models' runs
([`midterm_ms_e2_confusion_pairs.csv`](../results/tables/midterm_ms_e2_confusion_pairs.csv)); "A→B" means true A predicted B.

| class pair | clips per run | A→B / B→A |
|---|---:|---|
| Jet aircrafts – Propeller aircrafts | 3.2 | 1.8 / 1.4 |
| Bells – Music | 2.7 | 0.0 / 2.7 |
| Lawn mower brush cutter and olive shaker – Vehicle idling | 2.6 | 1.1 / 1.4 |
| Jet aircrafts – Vehicle pass-by | 2.4 | 0.3 / 2.1 |
| Vacuum cleaner fan and hairdryer – Workshop | 2.4 | 1.2 / 1.2 |
| Lawn mower – Vehicle pass-by | 2.2 | 0.1 / 2.1 |
| Vehicle idling – Workshop | 2.1 | 1.0 / 1.1 |

The confusions are again almost all inside the group of stationary, broadband engine / machine classes. The
exceptions are Bells ↔ Music (one-directional: Music is predicted as Bells) and, lower in the table, Music → Sirens
and alarms and Horn ↔ Voices. The listening checks of Part 2 (10 clips) found the labelled sound clearly audible in all of them.

## E3 (3 seeds). Does the CRNN's recurrence help on clips of different duration?

Difference in macro-F1, CRNN − `small_cnn`, per duration group
([`midterm_ms_e3_crnn_vs_cnn_by_duration.csv`](../results/tables/midterm_ms_e3_crnn_vs_cnn_by_duration.csv)).

| group | n | seed 42 | seed 43 | seed 44 | mean ± std |
|---|---:|---:|---:|---:|---:|
| short (≤ 5 s) | 309 | +0.020 | −0.036 | −0.009 | −0.008 ± 0.028 |
| medium (5–15 s) | 140 | −0.006 | +0.067 | +0.003 | +0.021 ± 0.040 |
| long (> 15 s) | 309 | +0.030 | +0.026 | −0.005 | +0.017 ± 0.019 |

**Answer: no consistent difference.** The sign changes with the seed in the short and medium groups, and the mean is
within one std of zero in all three. The single-run finding that the CRNN is worse on medium and long clips (−0.117,
−0.084) does not hold; it came from the early-stopped Colab CRNN. The confounds listed in Part 2 (the CRNN also differs
in its convolutional front end; duration is mixed with class; all models see only the centre 5 s) still apply, so even
a positive difference could not be attributed to the GRU.

## E4 (3 seeds). Which architecture has the best performance–cost balance?

Latency: batch 1 on the RTX 2060 SUPER, mean over seeds ([`rq4_pareto.csv`](../results/tables/rq4_pareto.csv)).
The Colab T4 latencies in Part 2 are not comparable.

| model | test F1 | params | MACs / clip | latency (ms) | train time (min) | on the Pareto front (clean F1 vs.) |
|---|---:|---:|---:|---:|---:|---|
| `small_cnn` | 0.876 | **0.39 M** | 198 M | **1.18** | 3.1 | latency, params |
| `mobilenetv3s` | **0.901** | 1.54 M | **18 M** | 6.88 | 3.7 | latency, MACs, params |
| `crnn` | 0.875 | 0.87 M | 487 M | 1.47 | 3.0 | none |

**Answer: no single winner; it depends on the cost you care about.**
- `mobilenetv3s` has the best mean F1 and the fewest MACs, but it is the slowest on a GPU at batch 1 (depthwise
  convolutions). MACs are the better proxy for a CPU or embedded device, which was not measured.
- `small_cnn` is the fastest and smallest, at about 0.025 lower F1 (about one std of `mobilenetv3s`).
- `crnn` is dominated: it is no more accurate than `small_cnn` and slower and larger in MACs.
- This reverses the Colab single-run answer ("`small_cnn`"), which came from one seed and the early-stopped CRNN.

# Part 2. Original Midterm answers (single Colab run, seed 42)

## E1. Which classes are recognised best and worst?

"Recognition accuracy" of a class is its recall (the diagonal of the row-normalised confusion matrix). Classes are
ranked by F1 averaged over the three models ([`midterm_e1_class_ranking.csv`](../results/tables/midterm_e1_class_ranking.csv);
precision / recall / F1 per model: [`midterm_e1_per_class.csv`](../results/tables/midterm_e1_per_class.csv)).

| | class | n test | F1 small_cnn | F1 mobilenetv3s | F1 crnn | mean F1 | mean recall |
|---|---|---:|---:|---:|---:|---:|---:|
| best | Cicadas and crickets | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| | Voices | 285 | 0.998 | 0.993 | 0.993 | 0.995 | 0.996 |
| | Music | 150 | 0.990 | 0.973 | 0.966 | 0.976 | 0.964 |
| | Wind turbine | 15 | 0.968 | 0.968 | 0.968 | 0.968 | 1.000 |
| worst | Jet aircrafts | 15 | 0.774 | 0.615 | 0.606 | 0.665 | 0.667 |
| | Vacuum cleaner fan and hairdryer | 15 | 0.690 | 0.714 | 0.545 | 0.650 | 0.644 |
| | Train | 11 | 0.583 | 0.545 | 0.609 | 0.579 | 0.606 |

**Answer.**
- **Best:** Cicadas and crickets, Voices, Music and Wind turbine (mean F1 ≥ 0.97).
- **Worst:** Train, Vacuum cleaner fan and hairdryer, Jet aircrafts, Vehicle idling and Lawn mower brush cutter and
  olive shaker (mean F1 0.58–0.73).

The weak end is consistent across the three models. Five classes are among the six worst of every model: Jet
aircrafts, Lawn mower, Train, Vacuum/fan/hairdryer and Vehicle idling. The sixth is Vehicle pass-by for `small_cnn`,
Propeller aircrafts for `mobilenetv3s` and Bells for `crnn`. Apart from Bells, all of these are stationary, broadband
engine or machine sounds (see E2).

**Caveat.** Three of the four best classes have curation artefacts that a model can use as shortcuts (A8):
- Voices: 88 % of clips cut off at 11.025 kHz;
- Music: 72 % cut off at 11.025 kHz, and 96 % of clips are 30 s long;
- Wind turbine: 93 % of clips are 10 s long.

Their high scores therefore do not show that the acoustic content was learned.

## E2. Which classes are confused most?

Errors are counted per unordered class pair, in both directions
([`midterm_e2_confusion_pairs.csv`](../results/tables/midterm_e2_confusion_pairs.csv)). The table sums the three
models; "A→B" means true A predicted as B.

| class pair | small_cnn | mobilenetv3s | crnn | total | A→B / B→A |
|---|---:|---:|---:|---:|---|
| Vacuum cleaner fan and hairdryer – Workshop | 2 | 2 | 8 | 12 | 5 / 7 |
| Jet aircrafts – Propeller aircrafts | 2 | 4 | 5 | 11 | 9 / 2 |
| Lawn mower brush cutter and olive shaker – Vehicle idling | 3 | 2 | 6 | 11 | 9 / 2 |
| Jet aircrafts – Vehicle pass-by | 2 | 2 | 4 | 8 | 1 / 7 |
| Jet aircrafts – Train | 2 | 3 | 3 | 8 | 3 / 5 |
| Vacuum cleaner fan and hairdryer – Vehicle idling | 3 | 1 | 4 | 8 | 6 / 2 |
| Bells – Music | 1 | 2 | 4 | 7 | 0 / 7 |
| Vehicle idling – Workshop | 2 | 3 | 2 | 7 | 3 / 4 |

**Answer.** The most serious confusions are within one group of 8 stationary, broadband engine / machine classes:
- Jet, Propeller, Train, Vehicle pass-by, Vehicle idling, Lawn mower, Vacuum/fan/hairdryer and Workshop;
- this group accounts for 32 of 46 small_cnn errors, 33 of 57 mobilenetv3s errors and 44 of 74 crnn errors;
- the strongest single confusions are Jet → Propeller aircrafts and Lawn mower → Vehicle idling (9 clips each over
  the three models).

Outside this group, the largest confusions involve Music: Music → Bells (7) and Music → Sirens and alarms (4, made by
all three models).

**Link to the dataset analysis (A6).** Spectral similarity explains only part of the confusions:
- **Similar and confused.** The most similar pair in A6 is Jet–Propeller (cosine 0.992), which is also the
  second-most confused pair.
- **Similar but never confused.** Glass breaking–Workshop (0.982), Glass breaking–Horn (0.942) and Horn–Workshop
  (0.927) produce **no** errors in any model. These sounds are short and impulsive. The time-averaged spectrum used in
  A6 discards this temporal structure, but the 2-D log-Mel keeps it.
- **Where confusions happen.** They occur where both the spectrum *and* the temporal envelope are similar: long,
  steady sounds.

`midterm_e2_eda_overlap.csv` lists the cosine next to the error counts for all 231 pairs. It is regenerated only after
`scripts/eda.py` has written the full similarity matrix.

**Hard clips.**
- 49 test clips are misclassified by at least 2 of the 3 models, and 22 by all three
  ([`midterm_e2_hard_clips.csv`](../results/tables/midterm_e2_hard_clips.csv)).
- 29 of the 49 are longer than 15 s.
- The models see only the centre 5 s of each clip. For the seven cropped clips in the audit below, that window keeps
  14–56 % of the clip energy ([`midterm_e2_audio_audit.csv`](../results/tables/midterm_e2_audio_audit.csv)), so the defining
  event can be outside the window.
- For the three short clips in the audit, zero padding makes up 46–65 % of the input.

### Qualitative error analysis: listening checks

The 10 clips below were picked by hand from the hard clips, one per true class. The "measured" column comes from
the audit table. The listening columns were filled in by a group member who listened to the 5 s model window of all 10 clips
(files in `outputs/listening_check/`). Finding: in every one of the 10 clips the labelled sound is clearly audible,
so these errors are not label noise and not a defining event lost outside the window. Only this was recorded; the
listener did not note background sounds or a cause per clip, so any cause beyond that is still a hypothesis.

| clip | true → predicted (small_cnn / mobilenetv3s / crnn) | measured | heard in the 5 s window | likely cause |
|---|---|---|---|---|
| Crows-0022 | Crows → Thunder/gunshot / Birds / Voices | 2.6 s clip, 49 % zero padding | Listener: crows clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| Jet aircrafts-0003 | Jet → Train / Propeller / Train | centre 5 s of 8.5 s, 56 % of energy | Listener: a jet aircraft clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| Lawn mower-0007 | Lawn mower → Vehicle idling (all) | centre 5 s of 17.3 s, 32 % of energy | Listener: a lawn mower / brush cutter clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| Music-0004 | Music → Sirens and alarms (all) | centre 5 s of 30 s, 14 % of energy | Listener: music clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| Helicopters-0035 | Propeller → Train / Train / Vehicle idling | centre 5 s of 8.7 s, 51 % of energy | Listener: a helicopter clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| Train-0005 | Train → Jet / Propeller / Propeller | centre 5 s of 27.8 s, 23 % of energy | Listener: a train clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| fan-0019 | Vacuum/fan → Vehicle idling / Propeller / Vehicle idling | centre 5 s of 25.2 s, 40 % of energy | Listener: a fan clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| motorbike idling-0013 | Vehicle idling → Glass breaking / Glass breaking / Crows | centre 5 s of 12.1 s, 40 % of energy | Listener: an idling motorbike clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| car pass-by-0011 | Vehicle pass-by → Thunder/gunshot (all) | 1.8 s clip, 65 % zero padding | Listener: a car passing by clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |
| Glass breaking-0001 | Glass breaking → Vehicle pass-by / correct / Voices | 2.7 s clip, 46 % zero padding | Listener: glass breaking clearly audible in the window | Not a label error or a missing event. Cause not identified by listening alone |

Waveform and log-Mel of the window the model saw: `results/figures/midterm_e2_audit_{Crows-0022, Jet aircrafts-0003,
Lawn mower-0007}.png`.

## E3. Does recurrent temporal modelling in the CRNN help over the plain 2-D CNN on clips of different duration?

Test clips are grouped by their **original** duration relative to the 5 s window:
- **short (≤ 5 s):** zero-padded;
- **medium (5–15 s)** and **long (> 15 s):** centre-cropped.

Macro-F1 is computed over the classes present in each group. The paired outcome compares `crnn` with `small_cnn` clip
by clip, and the p-value is an exact McNemar test on the clips where the two disagree
([`midterm_e3_duration_metrics.csv`](../results/tables/midterm_e3_duration_metrics.csv),
[`midterm_e3_paired_outcomes.csv`](../results/tables/midterm_e3_paired_outcomes.csv),
[`midterm_e3_duration_comparison.png`](../results/figures/midterm_e3_duration_comparison.png)).

| group | n | classes | F1 small_cnn | F1 crnn | ΔF1 (CRNN − CNN) | only CNN correct | only CRNN correct | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short (≤ 5 s) | 309 | 13 | 0.895 | 0.892 | −0.004 | 4 | 3 | 1.00 |
| medium (5–15 s) | 140 | 18 | 0.872 | 0.754 | −0.117 | 12 | 4 | 0.077 |
| long (> 15 s) | 309 | 20 | 0.860 | 0.776 | −0.084 | 25 | 6 | 0.0009 |

**Answer: no measurable improvement in this setup.**
- **Short clips:** the CRNN is equal to the CNN.
- **Medium and long clips:** it is worse. The gap is clear for long clips (25 vs 6 discordant clips) and borderline
  for medium clips.

This does **not** show that recurrence is useless, for four reasons:

1. **The architectures differ in more than the GRU.**
   - The CRNN has its own convolutional front end: 487 M vs 198 M MACs, pooling mainly in frequency.
   - It stopped early at epoch 20 (best epoch 12), while `small_cnn` ran 50 epochs (best 45).
   - The comparison is therefore confounded. A controlled variant, the `small_cnn` trunk plus a BiGRU, is planned for
     the Final ([final_rq_proposal.md](final_rq_proposal.md)).
2. **Duration is confounded with class.**
   - The short group is 231 / 309 Voices.
   - All 150 Music clips are in the long group.
   - Each group therefore mixes a duration effect with a class-composition effect
     ([`midterm_e3_duration_class_counts.csv`](../results/tables/midterm_e3_duration_class_counts.csv)).
3. **Every model sees only 5 s.** For long clips the GRU cannot use more context than the CNN: both see the same
   centre crop. Original duration mostly measures how much of the clip was *discarded*.
4. **Single seed.** McNemar treats the test clips as the only source of variation and ignores training randomness,
   so the p-values are optimistic.

## E4. Which architecture has the best performance–cost balance?

Pareto front over (cost, test macro-F1) for each cost measure
([`midterm_e4_tradeoff.csv`](../results/tables/midterm_e4_tradeoff.csv),
[`midterm_e4_tradeoff.png`](../results/figures/midterm_e4_tradeoff.png)).

| model | val F1 | test F1 | params | MACs / clip | latency (ms, T4, batch 1) | train time (min) | on the Pareto front for |
|---|---:|---:|---:|---:|---:|---:|---|
| `small_cnn` | 0.856 | **0.884** | **0.39 M** | 198 M | **1.32** | 5.0 | latency, MACs, params |
| `mobilenetv3s` | **0.882** | 0.853 | 1.54 M | **18 M** | 5.78 | 4.0 | MACs |
| `crnn` | 0.803 | 0.815 | 0.87 M | 487 M | 2.19 | 2.1 | none |

**Answer: `small_cnn`.**
- It has the highest test macro-F1 and the lowest latency and parameter count, so it is Pareto-optimal on every cost
  measure.
- `mobilenetv3s` needs 11× fewer MACs but is the slowest on the T4 (5.8 ms). Its depthwise convolutions do not turn
  the low MAC count into speed at batch size 1 on a GPU. MACs are the better proxy for a CPU or an embedded device,
  where MobileNet's advantage may hold, but this was not measured.
- `crnn` is dominated on every measure.

**Caveats.**
- **The ranking of `small_cnn` and `mobilenetv3s` is not settled.**
  - On validation, MobileNet is better (0.882 vs 0.856); on test, `small_cnn` is better (0.884 vs 0.853).
  - With one seed, the gap between them is within the uncertainty.
  - The front above uses test F1 because it reports the final models; it was not used to choose them.
- **Training time is not comparable.** It reflects early stopping (crnn ran 20 epochs, `small_cnn` 50), not the cost
  per epoch.

## Summary of open items before the presentation

- Seeds 43/44 are done: see Part 1.
- Fill in the listening columns of the E2 table.
- Run `scripts/eda.py` once to regenerate `midterm_e2_eda_overlap.csv` from code.
