# データ取り込み・前処理(IO標準化)

## 1.目的と範囲

- YAIBAのCSV/JSONログを標準化し、後工程(グラフ / 動画 / UI)が共通して利用できるDataFrameに変換。
- 他工程の要求に応じて、必要なDataFrameや処理機能を提供する。

## 2.成果物

- 修正中

## 3.入力仕様

- ログファイル(str)  
  ログファイルのPATH。  
  必須引数。  
  - VRChatログファイル(.txt / .log)
    - 参加,離脱ログ構造
      ```text
      yyyy.mm.dd HH:MM:DD Debug [Behaviour] OnPlayerJoinComplete 名前
      yyyy.mm.dd HH:MM:DD Debug [Behaviour] OnPlayerLeft 名前
      ```
    - 座標ログ構造  
      文字の定義は
      ```text
      L  = (x, y, z) [meter]  ... 位置
      R  = (θ, φ, ρ) [degree] ... 回転 (0 <= θ, φ, ρ < 360)
      ΔL = (x, y, z) [meter]  ... 変位
      θ:ピッチ, φ:ヨー, ρ:ロール
      ```
      とし、
      ```text
      timestamp Debug - [Player Position]user_id,user_name,Lx,Ly,Lz,Rθ,Rφ,Rρ,ΔLx,ΔLy,ΔLz,isVR
      ```
      ```text
      生データ例
      yyyy.mm.dd HH:MM:DD Debug - [Player Position]1,"名前",0,0,0,0,0,0,0,0,0,True
      ```
    **※YAIBAによるVRChatログバージョン1.0.0のログ解釈に不具合有り。要修正。**
  - YAIBA出力ログ(.csv)
    - 参加,離脱ログ構造
      ```text
      なし(YAIBAでCSV出力をすると座標以外のデータが排除される)
      ```
    - 座標ログ構造
      ```text
      timestamp,player_id,user_name,pseudo_user_name,location_x,location_y,location_z,rotation_1,rotation_2,rotation_3,location_dx,location_dy,location_dz,is_vr
      ```
  - YAIBA出力ログ(.json)
    - 参加,離脱ログ構造
      ```text
      {"timestamp": 時刻, "user_name": "名前, "pseudo_user_name": "仮名", "type_id": "vrc/player_join"}
      {"timestamp": 時刻, "user_name": "名前", "pseudo_user_name": "仮名", "type_id": "vrc/player_left"}
      ```
    - 座標ログ構造
      ```text
      {"timestamp": 時刻, "player_id": 番号, "user_name": "名前", "pseudo_user_name": "仮名", "location_x": X座標, "location_y": Y座標, "location_z": Z座標, "rotation_1": ピッチ, "rotation_2": ヨー, "rotation_3": ロール, "location_dx": X変位, "location_dy": Y変位, "location_dz": Z変位, "is_vr": VR or Desktop, "type_id": "yaiba/player_position"}
      ```
- 秒粒度(int)  
  positionに保持するデータのインターバル設定値。  
  初期値は1秒、入力範囲は1~3600秒。  
- 匿名化フラグ(bool)  
  YAIBAによるデータ解釈処理でユーザー名を匿名化する場合にTrueを入力する。  
  匿名化処理はYAIBAの標準機能を利用する。  
  初期値はTrue。
- 時刻基準点(datetime)  
  YAIBAの出力に含まれるtimestampはローカルタイム(設定としてはTimeZone=UTC)なので、  
  実際のイベント開始時刻を与えることで補正するために利用する。  
  最初のsecondとの差をtimedeltaオブジェクトとして保持し、全てのレコードを補正する。  
  値がNoneの場合は補正しない。  
  適用するtimestampは、positionとattendanceの両方。  
  初期値はNone。  

## 4.出力仕様

### Must仕様

座標データ / 参加, 離脱データ / 各種データを持ったオブジェクト(LogDataクラス)を返す。  
※値はプライベートにし、getterによるアクセスに限定する。(値の書き換えを原則禁止とする)  

- LogDataクラス  
  後工程で用いるデータをすべて保持する。  
  
  | インスタンス変数名  |             型              |       説明        |
  |:----------:|:--------------------------:|:---------------:|
  |  position  |      pandas.DataFrame      |      座標データ      |
  | attendance | Optional[pandas.DataFrame] |    参加,離脱データ     |
  |    area    |    Areaオブジェクト or float     | マップ境界データ(X,Y,Z) |
  |            |                            |

  | メソッド名 |       説明       |
  |:-----:|:--------------:|
  | save  | 処理結果を保存する処理を行う |  

  CSVファイルを読み込んだ場合は、参加,離脱データが存在しないのでNoneが入る。  
  areaデータを非オブジェクトで保持する場合は、x_min~z_maxまでをインスタンス変数としてLogDataに実装。  
  x_min ~ z_maxの詳細については「**4.出力仕様, Areaクラス**」を確認のこと。  
  **※その他のデータ等については、他工程からの要求に応じる形で追加する。**  
- position(座標データ / pandas.DataFrame)  
  ユーザーの位置に関するデータをPandasで保持する。  
  
  |    カラム名     |        型        |   単位   |        説明        |        データ例         |
  |:-----------:|:---------------:|:------:|:----------------:|:-------------------:|
  |   second    |    datetime     | 秒(UTC) |  ログデータが取得された時刻   | 2025-05-17 13:16:55 |
  |   user_id   |       int       |   -    | 各ユーザーに自動付与されるID  |         148         |
  |  user_name  |       str       |   -    |     ユーザーの名前      |          -          |
  | location_x  |      float      |  メートル  |   ユーザーの位置(X座標)   |      -15.30132      |
  | location_y  | Optional[float] |  メートル  |   ユーザーの位置(Y座標)   |      1.038264       |
  | location_z  |      float      |  メートル  |   ユーザーの位置(Z座標)   |      5.293732       |
  | rotation_1  |      float      |   度    | ユーザーの視点位置(X:ピッチ) |      359.8868       |
  | rotation_2  |      float      |   度    | ユーザーの視点位置(Y:ヨー)  |       358.99        |
  | rotation_3  |      float      |   度    | ユーザーの視点位置(Z:ロール) |      359.8888       |
  | location_dx | Optional[float] |  メートル  |  ユーザーの位置変化(X座標)  |     0.003905837     |
  | location_dy | Optional[float] |  メートル  |  ユーザーの位置変化(Y座標)  |      -1.326468      |
  | location_dz | Optional[float] |  メートル  |  ユーザーの位置変化(Z座標)  |    -0.001761862     |
  |    is_vr    |      bool       |   -    |  ユーザーがVRモードであるか  |        True         |
  |  event_day  |    datetime     | 日(JST) |  イベント開催日(ログ取得日)  |     2025-05-17      |
  |  is_error   |      bool       |   -    |  レコードに不備が含まれるか   |        False        |

  location_dx / location_dy / location_dzは秒粒度が1の場合、VRChatログデータをそのまま採用、  
  1を超える場合は計算値を採用。  
  VRChatログバージョン1.0.0未満の場合、location_y / location_dx / location_dy / location_dzにはNoneが入る。  
  secondはUTC、event_dayはJSTを基準とする。  
- attendance(参加,離脱データ / pandas.DataFrame)  
  ユーザーがワールドに参加,離脱した時刻を保持する。  
  
  |   カラム名   |    型     |   単位   |          説明          |        データ例         |
  |:--------:|:--------:|:------:|:--------------------:|:-------------------:|
  |  second  | datetime | 秒(UTC) |   join / leftした時刻    | 2025-05-17 13:16:55 |
  |  action  |   str    |   -    | join / leftを識別する文字列  |       "join"        |
  | user_id  |   int    |   -    | join / leftしたユーザーのID |         148         |
  | is_error |   bool   |   -    |    レコードに不備が含まれるか     |        False        |

  secondはUTCを基準とする。  
- Areaクラス(マップ境界データ)  
  ユーザーの位置情報(position)から最大値と最小値を出すことで、マップの領域(境界)を定める。
  
  | インスタンス変数名 |   型   |       説明       |
  |:---------:|:-----:|:--------------:|
  |   x_min   | float | マップ境界データ(X最小値) |
  |   x_max   | float | マップ境界データ(X最大値) |
  |   y_min   | float | マップ境界データ(Y最小値) |
  |   y_max   | float | マップ境界データ(Y最大値) |
  |   z_min   | float | マップ境界データ(Z最小値) |
  |   z_max   | float | マップ境界データ(Z最大値) |

### Want仕様(次期以降開発予定)

- クォータニオンの導入  
  オイラー角をそのまま用いて時間微分を行うと、正確に計算できなくなることが想定される。  
  視点移動角の変化を正しく計算することを目的として、クォータニオンを追加する。  
  ```text
  クォータニオン q = (w, x, y, z)
  ```
- position(座標データ / pandas.DataFrame)  

  |    カラム名     |        型        |   単位   |        説明        |        データ例         |
  |:-----------:|:---------------:|:------:|:----------------:|:-------------------:|
  |   second    |    datetime     | 秒(UTC) |  ログデータが取得された時刻   | 2025-05-17 13:16:55 |
  |   user_id   |       int       |   -    | 各ユーザーに自動付与されるID  |         148         |
  |  user_name  |       str       |   -    |     ユーザーの名前      |          -          |
  | location_x  |      float      |  メートル  |   ユーザーの位置(X座標)   |      -15.30132      |
  | location_y  | Optional[float] |  メートル  |   ユーザーの位置(Y座標)   |      1.038264       |
  | location_z  |      float      |  メートル  |   ユーザーの位置(Z座標)   |      5.293732       |
  | rotation_1  |      float      |   度    | ユーザーの視点位置(X:ピッチ) |      359.8868       |
  | rotation_2  |      float      |   度    | ユーザーの視点位置(Y:ヨー)  |       358.99        |
  | rotation_3  |      float      |   度    | ユーザーの視点位置(Z:ロール) |      359.8888       |
  | location_dx | Optional[float] |  メートル  |  ユーザーの位置変化(X座標)  |     0.003905837     |
  | location_dy | Optional[float] |  メートル  |  ユーザーの位置変化(Y座標)  |      -1.326468      |
  | location_dz | Optional[float] |  メートル  |  ユーザーの位置変化(Z座標)  |    -0.001761862     |
  |    is_vr    |      bool       |   -    |  ユーザーがVRモードであるか  |        True         |
  |  event_day  |    datetime     | 日(JST) |  イベント開催日(ログ取得日)  |     2025-05-17      |
  |    quat     |   Quatオブジェクト    |   -    |   レコードのクォータニオン   |          -          |
  |  is_error   |      bool       |   -    |  レコードに不備が含まれるか   |        False        |

## 5.処理フロー(概要)

### 全体フロー
**画像を挿入**

### positionデータフロー
**画像を挿入**

### attendanceデータフロー
**画像を挿入**

### Areaデータフロー
**画像を挿入**

## 6.パラメータ一覧
パラメータの詳細は「**3.入力仕様**」を確認のこと。

- ログファイル(str) : 必須引数
- 秒粒度(int) : 任意引数
- 匿名化フラグ(bool) : 任意引数
- 時刻基準点(datetime) : 任意引数

## 7.例外・エラー処理方針

- YAIBAではtry / raiseがほとんど使われていないので、YAIBAでの処理を一つのtry節で囲み、一括で投げる。
- YAIBAの処理でエラーが発生した場合、ログ解釈に失敗しているので処理を停止させる必要がある。
- loader関数の引数が不正であれば処理を停止させる。
  - 投げるエラーはValueErrorとする。
  - 具体的なテスト対象と内容は「**8.性能・品質要件**」を確認のこと。
- ログファイルの存在が確認できない場合は処理を停止させる。
- DataFrame化処理でレコードに不備がある場合は、is_error値をTrueにして処理は続行する。
  - 具体的なテスト対象と内容は「**8.性能・品質要件**」を確認のこと。

## 8.性能・品質要件

- 出力仕様に定めるフォーマットを満たしていること。
  1. 各データの型が仕様と一致していること。
  2. 各値が以下の条件を満たしていること。
     - LogDataクラス
       - ログファイル : str型でPATHを指定すること。
       - ログファイル : 指定されたPATHにファイルが存在すること。
       - 秒粒度 : int型で1以上3600以下の範囲であること。
       - 匿名化フラグ : bool型であること。
     - position
       - second : TimeZoneがUTCであること。
       - event_day : TimeZoneがJSTであること。
       - rotation : 値が0以上360未満であること。
       - location_y, location_dx, location_dy, location_dz : VRChatログバージョン1未満はNoneが入ること。
       - location_dx, location_dy, location_dz : 秒粒度が1の場合はVRChatログデータをそのまま使うこと。
       - location_dx, location_dy, location_dz : 秒粒度が1を超える場合は前回レコードとの差を使うこと。
     - attendance
       - second : TimeZoneがUTCであること。  
         positionのsecondとはレコードがそもそも違うため、必ずpositionとは別で確認すること。

## 9.検証方法

1. 型の評価はPython標準のisinstance関数を用いること。
2. Noneの評価はis演算子を用いること。
3. 値を検証する対象と内容は「**8.性能・品質要件**」を確認のこと。
