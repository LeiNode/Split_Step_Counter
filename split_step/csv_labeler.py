"""
CSVLabeler — adds a 'label' column to a single pose CSV recorded with
pose_csv_recorder.py, using the hand-entered SPLIT_STEP_RANGES below.

Label is 1 when timestamp_ms falls inside any of the defined split-step
ranges (boundaries inclusive), 0 otherwise. For batches of pre-cut clips use
build_labeled_dataset.py instead, which reads the ranges from clip filenames.

Usage
-----
    python -m split_step.csv_labeler assets/csv_files/pose1.csv [--out OUT.csv]

    labeler = CSVLabeler("assets/csv_files/pose1.csv")
    labeler.label()                                   # writes assets/csv_files/pose1_labeled.csv
    labeler.label("assets/csv_files/my_output.csv")   # custom output path
"""

import math
from pathlib import Path

import pandas as pd

# (start_ms, end_ms) — both boundaries inclusive. Hand-labeled for pose1.csv.
SPLIT_STEP_RANGES = [
    (566.667, 766.667),
    (2082.667, 2149.333),
    (4966.667, 5233.333),
    (7233.333, 7600),
    (9600, 9766.667),
    (16700, 16933.333),
    (18966.667, 19166.667),
    (23533.333, 23800),
    (29900, 30166.667),
    (32166.667, 32433.333),
    (32433.333, 34833.333),
    (36933.333, 37166.667),
    (39233.333, 39433.333),
    (41733.333, 41966.667),
    (46533.333, 46766.667),
    (53333.333, 53600),
    (55833.333, 56066.667),
    (61366.667, 61733.333),
    (63366.667, 63566.667),
    (73600, 73833.333),
    (79833.333, 80100),
    (82100, 82366.667),
    (84633.333, 84866.667),
    (91266.667, 91500),
    (93533.333, 93833.333),
    (96166.667, 96466.667),
    (101000, 101200),
    (105700, 105966.667),
    (108033.333, 108333.333),
    (111166.667, 111433.333),
    (113633.333, 113966.667),
    (116000, 116266.667),
    (119666.667, 119866.667),
    (124533.333, 125000),
    (134366.667, 134666.667),
    (138566.667, 139033.333),
    (143100, 143566.667),
    (147966.667, 148400),
    (152333.333, 152733.333),
    (161633.333, 161933.333),
    (166600, 166900),
    (171166.667, 171500),
    (175600, 176100),
    (180266.667, 180566.667),
    (192900, 193233.333),
    (197600, 198100),
    (202200, 202633.333),
    (206933.333, 207400),
    (211833.333, 213200),
    (228466.667, 229300),
    (242166.667, 244033.333),
    (258833.333, 260400),
    (274233.333, 275866.667),
    (305366.667, 306366.667),
    (332233.333, 333300),
    (346466.667, 348366.667),
    (362033.333, 363766.667),
    (377100, 378600),
    (394100, 395866.667),
    (439100, 440933.333),
    (454266.667, 456100),
    (470100, 470766.667),
    (485266.667, 487266.667),
]


def in_any_range(timestamp_ms: float, ranges) -> int:
    """1 if timestamp_ms lies in any (start, end) range (inclusive, ms floored), else 0."""
    for start, end in ranges:
        if math.floor(start) <= timestamp_ms <= math.floor(end):
            return 1
    return 0


class CSVLabeler:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def _read(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path)
        if "timestamp_ms" not in df.columns:
            raise ValueError(f"'timestamp_ms' column not found in {self.input_path}")
        return df

    def feed_forward(self, output_path: str | None = None) -> pd.DataFrame:
        """
        Forward-fill rows where all landmark columns are empty (no pose detected).
        Any such row that already carries label=1 is reset to 0 — undetected
        frames cannot be confirmed split-steps.

        Note: the windowing pipeline (data_windowing/) deliberately drops empty
        rows instead; this is only for the legacy single-CSV flow.

        output_path — defaults to overwriting the input file in-place.
        Returns the processed DataFrame.
        """
        df = self._read()

        meta_cols     = {"timestamp_ms", "split_step_count", "label"}
        landmark_cols = [c for c in df.columns if c not in meta_cols]

        was_missing = df[landmark_cols].isna().all(axis=1)

        df[landmark_cols] = df[landmark_cols].ffill()

        zeroed = 0
        if "label" in df.columns:
            zeroed = int((was_missing & (df["label"] == 1)).sum())
            df.loc[was_missing, "label"] = 0

        df = df.dropna(subset=landmark_cols)

        if output_path is None:
            output_path = self.input_path

        df.to_csv(output_path, index=False)

        filled = int(was_missing.sum())
        print(f"Forward-filled {filled} missing rows — {zeroed} label(s) reset to 0")
        print(f"Saved → {output_path}")
        return df

    def label(self, output_path: str | None = None) -> pd.DataFrame:
        """
        Load the CSV, add a 'label' column, and save.

        output_path — defaults to <input_stem>_labeled.csv in the same folder.
        Returns the labeled DataFrame.
        """
        df = self._read()
        df["label"] = df["timestamp_ms"].apply(lambda t: in_any_range(t, SPLIT_STEP_RANGES))

        if output_path is None:
            p = Path(self.input_path)
            output_path = str(p.with_stem(p.stem + "_labeled"))

        df.to_csv(output_path, index=False)

        total    = len(df)
        positive = df["label"].sum()
        print(f"Labeled {total} rows — {positive} positive ({positive/total*100:.1f}%)")
        print(f"Saved → {output_path}")
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Label a pose CSV with split-step annotations.")
    parser.add_argument("input",  help="Path to input CSV (e.g. assets/pose1.csv)")
    parser.add_argument("--out",  help="Output path (default: <input>_labeled.csv)")
    args = parser.parse_args()

    labeler = CSVLabeler(args.input)
    labeler.feed_forward()
    labeler.label(args.out)
