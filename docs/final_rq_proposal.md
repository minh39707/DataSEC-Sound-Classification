# Proposed research question for the Final

Draft for the Midterm deliverable "proposed Final research question". It builds on the dataset analysis
([dataset_analysis.md](dataset_analysis.md)) and the seed-42 baselines. The hypotheses below are **not yet tested**.

## RQ1 (required by the brief)

> How robust are environmental sound classifiers when the target sound is perturbed by realistic acoustic noise,
> temporal distortions, and spectral corruptions?

**Operational form.** How much does the test macro-F1 of `small_cnn`, `mobilenetv3s` and `crnn` drop when the fixed
clean test split is corrupted at evaluation time? Does the drop differ between architectures and between classes with
different acoustic structure?

## Design

**Kept fixed:**
- `splits/test.csv` and its labels;
- the Midterm checkpoints, 3 seeds per model;
- the runtime pipeline in `configs/default.yaml`.

No retraining for RQ1.

**Perturbations (test time only):** 4 types × 3 severities. The brief requires at least 3 types; the fourth is a
reserve in case one type turns out to be degenerate.

| Type | Applied to | Severities | Definition |
|---|---|---|---|
| Additive noise | 22.05 kHz waveform (`AudioClips.perturb`) | SNR +20 / +10 / 0 dB | Pink (1/f) Gaussian noise, scaled to the SNR of the 5 s window |
| Gain | waveform | −12 / −6 / +6 dB | Pure scaling, no clipping to [−1, 1]. In the log-Mel domain this is an additive constant |
| Time masking | waveform | 10 / 25 / 50 % of the window | Random contiguous segments set to 0 |
| Frequency masking | Mel power, **before** the log | 8 / 16 / 32 of 64 bands | One contiguous block of bands set to 0. Needs a spectrogram-level hook in `src/features.py`, because `perturb` acts on the waveform |

Pipeline-specific notes:

- **Gain stays meaningful.** The per-clip `top_db` floor is relative to each clip's peak and normalisation is global
  (one train-set mean/std), so a gain change shifts the whole log-Mel image rather than being normalised away. This
  is intended: absolute level correlates with the class through curation (dataset analysis A8), so gain tests
  whether the models learned that shortcut.
- **"Set to 0" is defined in power.** Zeroed time segments and zeroed Mel bands are set to 0 power before the log, so
  they end at the clip's floor (peak − 80 dB). This choice must be recorded in the config, because zeroing the
  log-Mel value instead gives a different input.
- **Paired corruptions.** Each corruption is seeded by `hash(sample_id, type, severity)`, so every model and seed sees
  the *same* corrupted inputs and model comparisons are paired.
- **Output files.** Predictions follow the existing naming,
  `results/predictions/{model}_seed{s}_test_{perturbation}_{severity}.csv`.

Mixing in real background recordings (street, rain) is an optional fifth type. It needs licensed background audio
that does not overlap DataSEC's classes.

## Reported results

**Model comparison:**
- $F1_{clean}$, $F1_{corrupted}$ and $\Delta F1 = F1_{clean} - F1_{corrupted}$, all **macro**-F1, mean ± std
  over seeds 42/43/44;
- degradation curves over severity for each perturbation type;
- paired bootstrap 95 % CIs for differences in $\Delta F1$ between models.

**Error analysis:**
- per-class $\Delta F1$ under each perturbation;
- confusion-matrix change between clean and corrupted test sets;
- clips that flip from correct to wrong, with listening checks;
- separate reporting of test clips shorter than 5 s (zero-padded) for time masking. Duration is a known shortcut
  (A2), and masking interacts with the padding.

## Hypotheses (from the dataset analysis and the Midterm results)

**H1. Additive noise.** It hurts most the low-frequency broadband group (Train, Vehicle pass-by/idling,
Jet/Propeller aircraft) and increases confusions inside it. This group already has the lowest clean F1 and the most
similar spectra (A6).

**H2. Time masking.** It hurts most the short impulsive classes (Gunshot, Horn, Dog). A 25–50 % mask can remove the
whole event: 99 % of Gunshot clips are shorter than 4 s.

**H3. Frequency masking.** It hurts harmonic or narrow-band classes (Sirens, Birds, Bells, Cicadas) more than
broadband noise-like classes.

**H4. Gain.** It degrades a model only if that model uses absolute level. If so, the most affected classes should be
those whose level is set by curation: Glass breaking and Voices are mostly peak-normalised, while Wind turbine and
Birds are quiet (A8).

**H5. CRNN vs CNN.** The CRNN tolerates time masking better than the CNN, because the GRU can aggregate evidence from
unmasked segments.

**Caveat on H5.** The current `crnn` and `small_cnn` use different convolutional front ends (487 M vs 198 M MACs) and
the CRNN stopped early at epoch 20, so a CNN–CRNN difference cannot be attributed to recurrence alone. A controlled
variant (the `small_cnn` trunk plus a BiGRU) would isolate it; this is deferred until seeds 43/44 are done.

## Proposed extension

- **First: Extension 3 / RQ4 (accuracy–efficiency Pareto).** Params, MACs, latency and training time are already
  logged per run, so the extra cost is only to place them next to the clean and corrupted macro-F1 from RQ1.
- **If time allows: Extension 1 / RQ2 (SpecAugment).** It requires retraining. Results must separate perturbations
  seen in training (time/frequency masking) from unseen ones (additive noise, gain); otherwise the benefit is
  overstated.

## Risks and limitations

- **Duration shortcut.** Zero padding exposes clip duration (duration-only macro-F1 0.272, A2). Time-masking
  results are therefore split by clip length.
- **Remaining leakage.** Two test clips share audio with train clips (A8). The effect on clean macro-F1 is
  ≤ 0.0014, and $\Delta F1$ is affected even less, because the same test set is used for both clean and corrupted
  scores.
- **Single seed so far.** All current results use seed 42. Model rankings and per-class claims need seeds 43/44.
