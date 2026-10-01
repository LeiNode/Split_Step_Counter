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
- **Hard-negative verification**: never empirically confirmed whether
  `labeled_training_set_2.csv`'s negative frames (serves, forehand drills,
  etc.) are genuine hard negatives rather than easy ones. Run the trained
  model over them and check the false-positive rate.
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

## Cleanup

- Delete the throwaway `cut_previews/` folder at the repo root once visual
  verification of scene-cut detection is fully wrapped up.
