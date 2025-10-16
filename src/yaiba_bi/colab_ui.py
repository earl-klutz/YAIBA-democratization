import os
import shutil
from typing import Iterator

# ディレクトリ定数
BASE = "YAIBA_data"
INPUT = f"{BASE}/input"
INTERMEDIATE = f"{BASE}/intermediate"
OUTPUT = f"{BASE}/output"


# 定数イテレータ
def __iters() -> Iterator[str]:
    yield INPUT
    yield INTERMEDIATE
    yield OUTPUT


# ディレクトリ初期設定
def initialize() -> None:
    for path in __iters():
        os.makedirs(path, exist_ok=True)


# ディレクトリ後処理
def finalize() -> None:
    shutil.rmtree(BASE)
