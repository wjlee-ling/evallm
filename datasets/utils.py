import re
import os
import pandas as pd
from glob import glob
from pathlib import Path


def concatenate_columns(row, columns, sep="\t"):
    return sep.join([str(row[col]) for col in columns])


def load_from_path(path) -> list:
    path = Path(path)
    if path.is_dir():
        # Collect base names of all '.translated.csv' files
        translated_variants = {
            p.stem.replace(".translated", ""): p
            for p in path.iterdir()
            if p.is_file() and p.name.endswith((".translated.csv", ".translated.xlsx"))
        }

        # Select files that do not have a '.translated.csv' variant
        return [
            p
            for p in path.iterdir()
            if p.is_file()
            and p.stem not in translated_variants
            and not p.name.endswith((".translated.csv", ".translated.xlsx"))
        ]
    return [path]


def combine_columns(df, old_cols: list, new_col: str):
    df[new_col] = df[old_cols].apply(lambda x: "\n".join(x.astype(str)), axis=1)
    df.drop(columns=old_cols, inplace=True)
    return df


# df = pd.read_csv(
#     "/Users/lwj/workspace/evallm/mmlu/Test/mmlu_test_1_1553.translated.csv", header=0
# )
# df = combine_columns(df, ["A1", "A2", "A3", "A4"], "A")
# df.to_csv("mmlu_test_1_1553.combined.translated.csv", index=False)

# def combine_csv_files(prefixes, input_dir=".", output_dir="."):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for prefix in prefixes:
#         # Process files that do not end with '.translated.csv'
#         normal_files = glob(os.path.join(input_dir, f"{prefix}*.csv"))
#         normal_files = [f for f in normal_files if not f.endswith(".translated.csv")]

#         if normal_files:
#             combined_normal_df = pd.concat(
#                 [pd.read_csv(f) for f in normal_files], ignore_index=True
#             )
#             combined_normal_df.to_csv(
#                 os.path.join(output_dir, f"{prefix}_combined.csv"), index=False
#             )

#         # Process files that end with '.translated.csv'
#         translated_files = glob(os.path.join(input_dir, f"{prefix}*.translated.csv"))

#         if translated_files:
#             combined_translated_df = pd.concat(
#                 [pd.read_csv(f) for f in translated_files], ignore_index=True
#             )
#             combined_translated_df.to_csv(
#                 os.path.join(output_dir, f"{prefix}_translated_combined.csv"),
#                 index=False,
#             )


# # Example usage
# prefixes = [
#     "abstract",
#     "anatomy",
#     "astronomy",
#     "business",
#     "clinical",
#     "college",
#     "computer",
#     "conceptual",
#     "econometrics",
#     "electrical",
#     "elementary",
#     "formal",
#     "global",
#     "high_school",
#     "human",
#     "international",
#     "jurisprudence",
#     "logical",
#     "machine",
#     "management",
#     "marketing",
#     "medical",
#     "miscellaneous",
#     "moral",
#     "nutrition",
#     "philosophy",
#     "prehistory",
#     "professional",
#     "public",
#     "security",
#     "sociology",
#     "us_foreign",
#     "virology",
#     "world",
# ]
# combine_csv_files(
#     prefixes, input_dir="~/Downloads/Dev", output_dir="~/Downloads/Dev_combined"
# )
