import glob
import re
from collections import defaultdict
from typing import Final, Iterable, NamedTuple

try:
    import orjson as json
except ImportError:
    import json

TOKEN_SIZE: Final[int] = 4
NGRAMS: Final[int] = 6
NGRAM_SIZE: Final[int] = TOKEN_SIZE * NGRAMS
START: Final[str] = "^" * TOKEN_SIZE
MODEL: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
MODEL_FILE: Final[str] = f"model_{TOKEN_SIZE}_{NGRAMS}.json"
INPUT_ENCODING: Final[str] = "iso-8859-15"


class NToken(NamedTuple):
    token: str
    ngram: str


def load_corpus_file(archivo: str) -> Iterable[str]:
    with open(archivo, "rt", encoding=INPUT_ENCODING) as f:
        text = f.read()

    pattern = r"<doc.*?>(.*?)</doc>"
    texts = re.finditer(pattern, text, re.DOTALL)
    for text in texts:
        yield text.group(1)


def tokens(text: str) -> Iterable[str]:
    for i in range(0, len(text), TOKEN_SIZE):
        yield text[i : i + TOKEN_SIZE]


def ngrams(text: str) -> Iterable[NToken]:
    ngram = START
    for token in tokens(text):
        if not token.strip() and not ngram[-TOKEN_SIZE:].strip():
            continue

        yield NToken(token=token, ngram=ngram)
        ngram = (ngram + token)[-NGRAM_SIZE:]


def process_file(file: str) -> None:
    print(f"Processing {file}")
    for text in load_corpus_file(file):
        for token, ngram in ngrams(text):
            MODEL[ngram][token] += 1


def save_model(fname: str) -> None:
    with open(fname, "wb") as f:
        f.write(json.dumps(MODEL))


def main() -> None:
    corpus_pattern = "./corpus/*"
    for i, file in enumerate(glob.glob(corpus_pattern)):
        process_file(file)

    save_model(MODEL_FILE)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    MODEL.clear()
    main()
