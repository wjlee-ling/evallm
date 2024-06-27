import re
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
