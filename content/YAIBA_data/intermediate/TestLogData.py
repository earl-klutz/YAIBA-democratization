from dataclasses import dataclass
import pandas as pd


@dataclass(slots=True)
class TestArea:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(slots=True)
class TestLogData:
    position: pd.DataFrame
    attendance: pd.DataFrame
    area: TestArea


def pkl_loader(path: str) -> TestLogData:
    with open(path, "rb") as f:
        return pd.read_pickle(f)


if __name__ == "__main__":
    data = pkl_loader("TestData.pkl")
    print(data.position)
