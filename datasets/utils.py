from pathlib import Path


def concatenate_columns(row, columns, sep="\t"):
    return sep.join([str(row[col]) for col in columns])


def load_from_path(path) -> list:
    path = Path(path)
    if path.is_dir():
        return [p for p in path.iterdir() if p.is_file()]
    return [path]
