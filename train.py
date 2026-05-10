#!/usr/bin/env python
import glob
import os
import argparse
from collections import defaultdict
from typing import Final, Iterable, NamedTuple, Literal

FILE_MODE: Literal["t", "b"] = "t"

try:
    import orjson as json
    FILE_MODE = "b"
except ImportError:
    import json

TOKEN_SIZE: Final[int] = 4
NGRAMS: Final[int] = 6
MODEL: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
INPUT_ENCODING: Final[str] = "utf-8"


class NToken(NamedTuple):
    token: str
    ngram: str


def load_corpus_file(archive: str) -> Iterable[str]:
    with open(archive, "rt", encoding=INPUT_ENCODING) as f:
        yield from f


def tokens(text: str, token_size: int) -> Iterable[str]:
    for i in range(0, len(text), token_size):
        yield text[i : i + token_size]


def ngrams(text: str, options) -> Iterable[NToken]:
    token_size = options.token_size
    ngram_size = options.ngrams * token_size
    ngram = "^" * token_size
    for token in tokens(text, token_size):
        if not token.strip() and not ngram[-token_size:].strip():
            continue

        yield NToken(token=token, ngram=ngram)
        ngram = (ngram + token)[-ngram_size:]


def process_file(file: str, options) -> None:
    print(f"Processing {file}")
    for text in load_corpus_file(file):
        for token, ngram in ngrams(text, options):
            MODEL.setdefault(ngram, {}).setdefault(token, 0)
            MODEL[ngram][token] += 1


def load_model(fname: str) -> None:
    global MODEL

    MODEL.clear()
    if not os.path.isfile(fname):
        return

    with open(fname, f"r{FILE_MODE}") as f:
        MODEL = json.loads(f.read())


def save_model(fname: str) -> None:
    print(f"saving model to {fname}")
    with open(fname, f"w{FILE_MODE}") as f:
        f.write(json.dumps(MODEL))


def main(options) -> None:
    model_file: Final[str] = f"model_{options.token_size}_{options.ngrams}.json"

    load_model(model_file)
    for i, file in enumerate(glob.glob(options.FOLDER, recursive=True)):
        process_file(file, options)

    save_model(model_file)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train from corpus at specific directory using chunk of letters.")
    parser.add_argument("--token-size", type=int, default=TOKEN_SIZE, help="Character chunk size", required=False)
    parser.add_argument("--ngrams", type=int, default=NGRAMS, help="Number of ngrams", required=False)
    parser.add_argument("FOLDER", help="folder pattern where text files are allocated")
    options = parser.parse_args()
    main(options)
