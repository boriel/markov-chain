import random
from collections import deque
from typing import Final

try:
    import orjson as json
except ImportError:
    import json

NGRAMS: Final[int] = 4
START: Final[str] = "^^"
MODEL_FILE: Final[str] = f"model_word_{NGRAMS}.json"


def load_model(fname: str) -> None:
    global MODEL
    with open(fname, "rb") as f:
        MODEL = json.loads(f.read())


def predict(ngram: deque) -> str:
    text = " ".join(ngram)
    if text not in MODEL:
        # print(f"{text} not in model")
        return ""

    return random.choices(
        population=list(MODEL[text].keys()), weights=list(MODEL[text].values()), k=1
    )[0]


def generate(start: str = START, length: int = 1000) -> str:
    text = ""
    current = deque(start.split(), maxlen=NGRAMS)
    for i in range(length):
        token = predict(current)
        if token == "":
            break
        text += f" {token}"
        current.append(token)
    return text


if __name__ == "__main__":
    load_model(MODEL_FILE)
    print(generate())
