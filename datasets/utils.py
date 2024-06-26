from dotenv import load_dotenv

load_dotenv()


def concatenate_columns(row, columns, sep="\t"):
    return sep.join([str(row[col]) for col in columns])
