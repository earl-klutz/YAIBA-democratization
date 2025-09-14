# YAIBA Loader 開発者向け説明書

## 概要
本リポジトリでは、VRChat/YAIBA ログを標準化・加工するための **YAIBA Loader** を提供しています。  
開発・サービス運用の用途に応じて、以下の 3 本の Python スクリプトを使用します。

---

## 1. サービス本体: `yaiba_loader.py`

### 役割
- サービスとしてローンチされる唯一のスクリプト。
- ユーザが指定したログファイルを読み込み、以下を処理:
  - 匿名化 (`--is-pseudo` / `--no-is-pseudo`)
  - リサンプリング (`--span`)
  - time_sync 補正 (`--time-sync`)
- 最終的に `ld.position` / `ld.attendance` を生成し、先頭数行を print します。  

> **event_day について**  
> `event_day` は「JSTの日付列」です。pandasの仕様上 `dtype: object` となりますが、**各要素は Python の `datetime.date` 型**です（＝設計書で定義された「日付列」の要件を満たします）。  

### 実行方法

#### Windows PowerShell
```powershell
cd src\engine\core
python yaiba_loader.py --user-path "C:\path\to\log.txt" --span 1 --is-pseudo --time-sync "2025-09-06 22:55:27"
```

#### Linux / Colab
```bash
cd /content/src/engine/core
python yaiba_loader.py --user-path "/content/path/log.txt" --span 1 --is-pseudo --time-sync "2025-09-06 22:55:27"
```

---

## 2. 開発者向け単体検証: `writecsv.py`（旧 validate_position_attendance.py）

### 役割
- 開発者が `yaiba_loader.load()` を呼び出して **position / attendance を CSV 保存**するための補助スクリプト。
- 保存先は `YAIBA_data/output/meta/valid/` 以下。  
※ `output/meta/valid/` は**開発・検証用の補助フォルダ**であり、成果物には含めません（設計書の `meta` 配下運用に準拠）。

### 実行方法

#### Windows PowerShell
```powershell
cd src\engine\core
python writecsv.py --user-path "C:\path\to\log.txt" --span 1 --is-pseudo --time-sync "2025-09-06 22:55:27"
```

#### Linux / Colab
```bash
cd /content/src/engine/core
python writecsv.py --user-path "/content/path/log.txt" --span 1 --is-pseudo --time-sync "2025-09-06 22:55:27"
```

### 出力
- `YAIBA_data/output/meta/valid/position_*.csv`
- `YAIBA_data/output/meta/valid/attendance_*.csv`（空ならスキップ）

---

## 3. 設計書準拠テスト: `test_yaiba_loader_full.py` ※現在はPullしていません。


### 役割
- 設計書の仕様に従い、YAIBA Loader の動作を網羅的に検証します。
- OK なら設計書の文言を引用し「問題なし」と表示。
- NG ならエラー内容を表示し、ログに残します。

### 実行方法

#### Windows PowerShell
```powershell
cd src\engine\core
python test_yaiba_loader_full.py --debug
# JSON レポートも保存する場合
python test_yaiba_loader_full.py --json "C:\path\to\validate_report.json" --debug
```

#### Linux / Colab
```bash
cd /content/src/engine/core
python test_yaiba_loader_full.py --debug
# JSON レポートも保存する場合
python test_yaiba_loader_full.py --json "/content/YAIBA_data/output/meta/valid/validate_report.json" --debug
```

### 出力
- コンソールにチェック結果を表示。
- ログは `YAIBA_data/output/meta/logs/` に保存。

---

## パラメータ一覧
（設計書「3.入力仕様」「6.パラメータ一覧」に準拠）

- **ログファイル (str)** : 必須  
  `--user-path` で指定  
- **秒粒度 (int)** : 任意  
  `--span` （1〜3600, default=1）  
- **匿名化フラグ (bool)** : 任意  
  `--is-pseudo`（デフォルト有効） / `--no-is-pseudo` で無効化  
- **時刻基準点 (datetime)** : 任意  
  `--time-sync "YYYY-MM-DD HH:MM:SS"`

---

## 補足
- `YAIBA_data/input` は処理後にクリーンアップされます（ただし元のローカル入力ファイルは削除されません）。  
- `event_day` は pandas の制約により `dtype: object` ですが、各要素は Python の `datetime.date` 型です。  

---

📌 この3本のスクリプトの役割分担:
- **サービス運用** → `yaiba_loader.py`  
- **開発者の検証** → `writecsv.py`  
- **設計書遵守確認** → `test_yaiba_loader_full.py`  
