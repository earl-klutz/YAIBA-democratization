from dataclasses import dataclass
from typing import Optional, Tuple
import json
from datetime import datetime,date
import numpy as np
import pandas as pd
import yaiba


# ================================
# Classes
# ================================
@dataclass(frozen=True)
class Area:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

class LogData:
    """
    標準化済みログの入れ物。
      - position: DataFrame
      - attendance: Optional[DataFrame]
      - area: Area
    値はプライベートに保持し、getter経由でアクセスする。
    """
    def __init__(self, position: pd.DataFrame,
                 attendance: Optional[pd.DataFrame],
                 area: Area) -> None:
        self._position = position
        self._attendance = attendance
        self._area = area

    # ---- getters ----
    def get_position(self) -> pd.DataFrame:
        return self._position

    def get_attendance(self) -> Optional[pd.DataFrame]:
        return self._attendance

    def get_area(self) -> Area:
        return self._area



# ================================
# Helpers
# ================================

schema_yaiba = {
    "position": {
        "keys": [
            "timestamp", "player_id", "pseudo_user_name",
            "location_x", "location_y", "location_z",
            "rotation_1", "rotation_2", "rotation_3",
            "velocity_x", "velocity_y", "velocity_z",
            "is_vr", "type_id",
        ],
        "types": [
            float, int, str,
            float, float, float,
            float, float, float,
            float, float, float,
            bool, str
        ],
    },
    "attendance": {
        "keys": ["timestamp", "pseudo_user_name", "type_id"],
        "types": [float, str, str],
    },
}

schema_intermediate = {
    "position": {
        "keys": [
            "second", "user_id", "user_name",
            "location_x", "location_y", "location_z",
            "rotation_1", "rotation_2", "rotation_3",
            "location_dx", "location_dy", "location_dz",
            "is_vr", "event_day", "is_error"
        ],
        "types": [
            datetime, int, str,
            float, float, float,
            float, float, float,
            float, float, float,
            bool, date, bool
        ]
    },
    "attendance": {
        "keys": [
            "second", "action", "user_id",
            "is_error"
        ],
        "types": [
            datetime, str, int,
            bool
        ]
    }
}



def load_session_log(log_file: str):
    """
    ファイル読み込み
    """
    try:
        with open(log_file, encoding="utf-8") as fp:
            session_log = yaiba.parse_vrchat_log(fp)
    except FileNotFoundError:
        raise ValueError(f"指定されたログファイルが存在しません: {log_file}")
    except PermissionError:
        raise ValueError(f"指定されたログファイルにアクセスできません: {log_file}")
    except Exception as e:
        raise ValueError(f"ログファイルの読み込みに失敗しました: {log_file}, 詳細: {e}")

    if session_log is None:
        raise ValueError("YAIBA パースに失敗しました。session_log が None です。")

    return session_log


def infer_sec_interval(df_pos: pd.DataFrame) -> int | None:
    """
    df_posから秒粒度(sec_interval)を推定する関数

    - player_idごとにtimestampを並べて、その差を調べる
    - もしすべての差が同じ整数ならその値を返す
    - そうでなければ None を返す
    """

    # すべての差を集めるリスト
    all_diffs = []

    # player_idごとに処理
    for pid,group in df_pos.groupby("player_id"):
        # timestampで並べ替え
        group = group.sort_values("timestamp")
        timestamps = group["timestamp"].tolist()

        # 隣同士の差を計算
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i-1]
            all_diffs.append(diff)

    if len(all_diffs) == 0:
        print("差分を計算できませんでした。")
        return None

    # 一意の値かどうか確認
    unique_vals = set(all_diffs)
    if len(unique_vals) == 1:
        sec = unique_vals.pop()
        print(f"秒粒度は {sec} 秒です。")
        return sec
    else:
        print("秒粒度が一意に定まりません。")
        print("候補:", unique_vals)
        return None

def _normalize_action(v: str) -> str:
    """attendance用のaction正規化"""
    if v == "vrc/player_join":
        return "join"
    elif v == "vrc/player_left":
        return "left"
    else:
        return "unknown"
    
def _isinstance_map(vals, types) -> bool:
    return isinstance(vals,types)


# ================================
# DataEng
# ================================


def yaiba2df(session_log, schema: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """session_log から position / attendance の2表を構築"""
    try:
        encoder = yaiba.JsonEncoder(options=None)
        data = json.loads(encoder.encode(session_log))
    except Exception as e:
        # YAIBA パースに失敗したら即停止
        raise ValueError(f"YAIBA ログのパースに失敗しました: {e}")

    # log_entries がなければ即停止
    if "log_entries" not in data:
        raise ValueError("YAIBA ログに log_entries が含まれていません。")

    entry = data["log_entries"]

    pos_records = []
    attendance_records = []


    pos_keys = schema["position"]["keys"]
    pos_types = schema["position"]["types"]
    attendance_keys = schema["attendance"]["keys"]
    attendance_types = schema["attendance"]["types"]

    for record in entry:
        
        type_id = record["type_id"]
        if type_id == "yaiba/player_position":
            L = []
            for key in pos_keys:
                L.append(record[key])
            ret = map(_isinstance_map,L,pos_types)
            is_error = False in ret
            L.append(is_error)
            pos_records.append(L)
            

        elif type_id in ["vrc/player_join", "vrc/player_left"]:
            L = []
            for key in attendance_keys:
                L.append(record[key])
            ret = map(_isinstance_map,L,attendance_types)
            is_error = False in ret
            L.append(is_error)
            attendance_records.append(L)

    df_pos = pd.DataFrame(pos_records,columns = pos_keys+["is_error"])
    df_event = pd.DataFrame(attendance_records,columns = attendance_keys+["is_error"])

    return df_pos, df_event


def GenerateIntermediate(df_pos: pd.DataFrame, df_event: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    - df_pos に second(UTC) と event_day(JST) を追加
    - df_event に second(UTC) を追加し、action列を join/left に正規化
    """

    # --- position 側 ---

    rename_map = {
        "player_id": "user_id",
        "pseudo_user_name": "user_name",
        "velocity_x": "location_dx",
        "velocity_y": "location_dy",
        "velocity_z": "location_dz",
        }

    if not df_pos.empty and "timestamp" in df_pos.columns:
        df_pos["second"] = pd.to_datetime(df_pos["timestamp"], unit="s")
        df_pos["second"] = df_pos["second"].dt.tz_localize("UTC")
        df_pos["event_day"] = df_pos["second"].dt.tz_convert("Asia/Tokyo").dt.date
        df_pos["second"] = df_pos["second"].dt.tz_localize(None) 
        df_pos = df_pos.rename(columns=rename_map)

    # --- attendance 側 ---
    rename_map = {
        "pseudo_user_name": "user_id",
        }
    if not df_event.empty and "timestamp" in df_event.columns:
        df_event["second"] = pd.to_datetime(df_event["timestamp"], unit="s")
        df_event["second"] = df_event["second"].dt.tz_localize("UTC") 
        df_event["second"] = df_event["second"].dt.tz_localize(None)
        df_event["action"] = df_event["type_id"].map(_normalize_action)
        df_event = df_event.drop(columns=["type_id"])
        df_event = df_event.rename(columns=rename_map)

    

    # --- schema 順でカラム整列（pandas 1行方式） ---
    if not df_pos.empty:
        desired = schema["position"]["keys"]
        cols = [c for c in desired if c in df_pos.columns]
        df_pos = df_pos[cols]

    if not df_event.empty:
        desired = schema["attendance"]["keys"]
        cols = [c for c in desired if c in df_event.columns]
        df_event = df_event[cols]

    return df_pos, df_event

def build_area(position: pd.DataFrame) -> Area:
    """position の location_x/y/z の範囲から Area を作成。"""
    if position is None or position.empty:
        return Area(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    x_min, x_max = np.nanmin(position["location_x"]), np.nanmax(position["location_x"])
    y_min, y_max = np.nanmin(position["location_y"]), np.nanmax(position["location_y"])
    z_min, z_max = np.nanmin(position["location_z"]), np.nanmax(position["location_z"])
    return Area(float(x_min), float(x_max),
                float(y_min), float(y_max),
                float(z_min), float(z_max))



# ================================
# main
# ================================
def main(log_file: str,
        sec_interval: int = 1,
        anonymize: bool = True,
        base_time: Optional[datetime] = None) -> LogData:
    
    # --- 専用関数で読み込み＋例外処理 ---
    session_log = load_session_log(log_file)

    # 2表を構築
    df_pos, df_event = yaiba2df(session_log,schema_yaiba)

    # 秒粒度の推定
    infer_sec_interval(df_pos)
    df_pos, df_event = GenerateIntermediate(df_pos, df_event,schema_intermediate)

    # Areaを算出し、LogDataに格納
    area = build_area(df_pos if not df_pos.empty else None)
    logdata = LogData(position=df_pos, attendance=df_event, area=area)

    return logdata


if __name__ == "__main__":
    log_file = r"C:\Users\tinyt\Downloads\output.txt"

    logdata = main(log_file, sec_interval=1, anonymize=True, base_time=None)
    position = logdata.get_position()
    attendance = logdata.get_attendance()
    area = logdata.get_area()

    # 動作確認（必要に応じて保存に置き換え可）
    print("Position (head):")
    print(position.head())
    print("\nAttendance (head):")
    print(attendance.head())
    print("\nArea:", area)

