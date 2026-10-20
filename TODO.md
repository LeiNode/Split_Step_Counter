# TODO

Tracked follow-up items that are explicitly out of scope for whatever's being
worked on right now, so they don't get lost.

## Model / data pipeline

- **Feature engineering revisit**: bilateral left/right symmetry features
  (e.g. ankle-height difference), leg-length normalization for
  camera-distance/body-size invariance, and making use of the per-landmark
  visibility scores already recorded in the CSVs but currently unused as
  model input. Deferred until after the stateful per-frame LSTM redesign is
  validated, so it's clear which change produced which improvement.
- **Hard-negative verification**: measured 2026-10-19 with
  `python -m split_step.eval_negatives` (deployed Run 8 checkpoint). The
  negatives ARE hard: even on these training frames the model calls 9.5% of
  `labeled_training_set_2.csv`'s 91.6k negative frames "in a hop" (~19 false
  split steps per minute of negative footage); 75 of 204 segments exceed 10%,
  the worst (e.g. `video9`, `video18`, `video5` segments) 25-43%. Held-out
  `test_set_labeled.csv`: 18.9% negative-frame FP rate, 12.3 false/min, 29.1%
  positive recall (matches the logged 0.291). Next step if prioritized: look
  at the worst segments to see which movements fool it (serves? shuffles?)
  and whether they're mislabelled.
- **Gradual fade/dissolve transitions aren't caught** by the scene-cut
  detector in `split_step/build_labeled_dataset.py` (`_CutDetector` only
  fires on abrupt hard cuts below the 0.7 correlation threshold; a
  multi-frame fade/dissolve can stay above threshold on every single
  adjacent-frame pair while still crossing between two different scenes).
  Confirmed present in at least one real clip. Low practical impact so far
  (cross-faded frames are usually filtered out anyway by the ankle/hip
  visibility gate), but not fixed.
- **Training batching performance**: v1 of the stateful LSTM redesign trains
  one continuous sub-sequence at a time (no padding/packing across
  sequences) for implementation simplicity. Revisit batched/parallel
  sequence training if CPU training time becomes a problem.

## Data collection

- **Keep gathering diverse training footage** — more matches, players,
  courts, and camera setups remains the primary lever for generalization,
  independent of any model/architecture change.
- **Camera-angle imbalance, confirmed 2026-10-01**: measured per-video
  shoulder lateral-spread vs. depth-spread across the training CSVs —
  `labeled_training_set_2.csv` (the bulk of the data, mostly YouTube "Court
  Level View" clips) is overwhelmingly front/back-angle footage despite
  several filenames saying "side view" (those broadcast shots are filmed
  down the court's center axis, not from the side). The near player (back to
  camera) dominates the surviving frames; the far player (front view) is
  small and often dropped by the visibility gate in
  `build_labeled_dataset.py`; genuine side view is a small minority
  (`video_1`/`video_3` in `labeled_training_set_1.csv`). User confirmed live
  via the GUI: model performs well on back view, noticeably worse on side
  and front view — matches the data composition. Next step if prioritized:
  record/label footage specifically from side and front angles (not more
  back-view "court level" clips) and retrain.

## Cleanup

- Delete the throwaway `cut_previews/` folder at the repo root once visual
  verification of scene-cut detection is fully wrapped up.
