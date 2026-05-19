"""
CSVLabeler — adds a 'label' column to a pose CSV.

Label is 1 when timestamp_ms falls inside any of the defined split-step
ranges (boundaries inclusive), 0 otherwise.

Usage
-----
    labeler = CSVLabeler("assets/pose1.csv")
    labeler.label()                          # writes assets/pose1_labeled.csv
    labeler.label("assets/my_output.csv")    # custom output path
"""

import math
import pandas as pd

# (start_ms, end_ms) — both boundaries inclusive
SPLIT_STEP_RANGES = [
    (2266.66666667,  2833.33333333),
    (20833.3333333,  21266.6666667),
    (23566.6666667,  24400.0),
    (26866.666,      27433.333),
    (29500.0,        29966.666),
    (35866.666,      36200.0),
    (38200.0,        38933.333),
    (41566.666,      42266.666),
    (48033.333,      48566.666),
    (50166.666,      50466.666),
    (52933.333,      53433.333),
    (55800.0,        56233.333),
    (58433.333,      58800.0),
    (61300.0,        61766.666),
    (63966.666,      64400.0),
    (66400.0,        66666.666),
    (73166.666,      73500.0),
    (76100.0,        76600.0),
    (83466.666,      83733.333),
    (88800.0,        89266.666),
    (92366.666,      92900.0),
    (96300.0,        96633.333),
    (105966.666,    106266.666),
    (115533.333,    115700.0),
    (117800.0,      118166.666),
    (120500.0,      120700.0),
    (122900.0,      123233.333),
    (125300.0,      125633.333),
    (130533.333,    130766.666),
    (131966.666,    132433.333),
    (141800.0,      142066.666),
    (144433.333,    144600.0),
    (154633.333,    155000.0),
    (157100.0,      157333.333),
    (158600.0,      159133.333),
    (167766.666,    168166.666),
    (170500.0,      170966.666),
    (182500.0,      182866.666),
    (185266.666,    185666.666),
    (188033.333,    188333.333),
    (190833.333,    191200.0),
    (197600.0,      197900.0),
    (201633.333,    201833.333),
    (204033.333,    204433.333),
    (206400.0,      206733.333),
    (212566.666,    212700.0),
    (218200.0,      218533.333),
    (220966.666,    221233.333),
    (223333.333,    223566.666),
    (228533.333,    228900.0),
    (238133.333,    238533.333),
    (240933.333,    241233.333),
    (243566.666,    244033.333),
    (248600.0,      248766.666),
    (250733.333,    251000.0),
    (267966.666,    268300.0),
    (270400.0,      270800.0),
    (273066.666,    273300.0),
    (284633.333,    284933.333),
    (286966.666,    287333.333),
    (294200.0,      294433.333),
    (296900.0,      297166.666),
    (299533.333,    299800.0),
    (302933.333,    303233.333),
    (308066.666,    308300.0),
    (310200.0,      310566.666),
    (312733.333,    313000.0),
    (315166.666,    315600.0),
    (324566.666,    324866.666),
    (327333.333,    327600.0),
    (341600.0,      341833.333),
    (344433.333,    344733.333),
    (346900.0,      347233.333),
    (351333.333,    351633.333),
    (353333.333,    353633.333),
    (359033.333,    359300.0),
    (372000.0,      372233.333),
    (380766.666,    381066.666),
    (388833.333,    389166.666),
    (396800.0,      397100.0),
    (399433.333,    399733.333),
    (402000.0,      402466.666),
    (407433.333,    407666.666),
    (414166.666,    414500.0),
    (416466.666,    416800.0),
    (419000.0,      419333.333),
    (421266.666,    421500.0),
    (424166.666,    424433.333),
    (431066.666,    431333.333),
    (436233.333,    436500.0),
    (438566.666,    438966.666),
    (445766.666,    446100.0),
    (451433.333,    451666.666),
    (454166.666,    454433.333),
    (473733.333,    474000.0),
    (476500.0,      476800.0),
    (488700.0,      489000.0),
    (497466.666,    497833.333),
    (500233.333,    500600.0),
    (507933.333,    508333.333),
    (513000.0,      513866.666),
    (519033.333,    519266.666),
    (524100.0,      524566.666),
    (535633.333,    535866.666),
    (543000.0,      543333.333),
    (549333.333,    549566.666),
    (570266.666,    570666.666),
    (572966.666,    573300.0),
    (575700.0,      575933.333),
    (581666.666,    581933.333),
    (584166.666,    584633.333),
    (586966.666,    587366.666),
    (589500.0,      589833.333),
    (592133.333,    592466.666),
    (595266.666,    595566.666),
    (598266.666,    598500.0),
]


def _in_any_range(ts: float) -> int:
    for start, end in SPLIT_STEP_RANGES:
        if math.floor(start) <= ts <= math.floor(end):
            return 1
    return 0


class CSVLabeler:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def feed_forward(self, output_path: str | None = None) -> pd.DataFrame:
        """
        Forward-fill rows where all landmark columns are empty (no pose detected).
        Any such row that already carries label=1 is reset to 0 — undetected
        frames cannot be confirmed split-steps.

        output_path — defaults to overwriting the input file in-place.
        Returns the processed DataFrame.
        """
        df = pd.read_csv(self.input_path)

        if "timestamp_ms" not in df.columns:
            raise ValueError(f"'timestamp_ms' column not found in {self.input_path}")

        meta_cols    = {"timestamp_ms", "split_step_count", "label"}
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
        df = pd.read_csv(self.input_path)

        if "timestamp_ms" not in df.columns:
            raise ValueError(f"'timestamp_ms' column not found in {self.input_path}")

        df["label"] = df["timestamp_ms"].apply(_in_any_range)

        if output_path is None:
            from pathlib import Path
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

    CSVLabeler(args.input).label(args.out)
