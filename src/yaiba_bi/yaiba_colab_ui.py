import os
from enum import Enum
from typing import Iterator


class DIRS(Enum):
    INPUT = "YAIBA_data/input"
    INTERMEDIATE = "YAIBA_data/intermediate"
    OUTPUT = "YAIBA_data/output"

    @classmethod
    def values(cls) -> Iterator[str]:
        for _dir in cls:
            yield _dir.value


def setup_dirs() -> None:
    for _dir in DIRS.values():
        os.makedirs(_dir, exist_ok=True)
