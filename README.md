# YAIBA Democratization Project

YAIBA Democratization は VRChat ギミック "YAIBA" が出力するログを誰でも扱えるようにするための解析ツール群です。ログを標準化したデータフレームに変換し、ヒートマップ・滞在時間ヒストグラム・軌跡動画などの可視化を行うことを目指しています。詳細な仕様は `docs/` フォルダの設計書を参照してください。

## Requirements
- Python 3.10 or later (developed with 3.12)
- pip

## Setup
1. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Data import & preprocessing
YAIBA や VRChat のログを標準形式 (pandas DataFrame) に変換します。

### 2. Visualization
変換済みデータを用いて以下の成果物を生成します。
- **ヒートマップ**: イベント会場内の滞留を2D heatmapで表示
- **滞在時間ヒストグラム**: 参加者の滞在時間分布を可視化
- **軌跡動画**: x–z 平面での動きを時系列アニメーションとして出力

各処理の詳細やパラメータは、
- `docs/設計書_データ取り込み・前処理（I-O標準化）.md`
- `docs/設計書_可視化（ヒートマップ）.md`
- `docs/設計書_可視化(ヒストグラム).md`
- `docs/設計書_可視化(動画).md`
- `docs/設計書_UIUX.md`
を参照してください。

## Tests
The project plans to use `pytest` for testing.
```bash
pytest
```
(Currently no test cases are bundled.)

## Directory Structure
- `docs/` – design documents and glossary
- `LICENSE` – MIT license

## License
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.

