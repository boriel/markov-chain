import random
from typing import Final

try:
    import orjson as json
except ImportError:
    import json

TOKEN_SIZE: Final[int] = 4
NGRAMS: Final[int] = 6
NGRAM_SIZE: Final[int] = TOKEN_SIZE * NGRAMS
START: Final[str] = "^" * TOKEN_SIZE
MODEL_FILE: Final[str] = f"model_{TOKEN_SIZE}_{NGRAMS}.json"


def load_model(fname: str) -> None:
    global MODEL
    with open(fname, "rb") as f:
        MODEL = json.loads(f.read())


def predict(text: str = START) -> str:
    text = text[-NGRAM_SIZE:]
    if text not in MODEL:
        # print(f"{text} not in model")
        return ""

    return random.choices(
        population=list(MODEL[text].keys()), weights=list(MODEL[text].values()), k=1
    )[0]


def generate(start: str = START, length: int = 1000) -> str:
    text = start
    for i in range(length):
        token = predict(text)
        if token == "":
            break
        text += token
    return text


if __name__ == "__main__":
    load_model(MODEL_FILE)
    print(generate())
