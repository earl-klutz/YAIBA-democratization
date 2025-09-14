# src/engine/core/yaiba_loader.py
# Requirements: pandas, numpy, pyyaml
from __future__ import annotations

import re
import csv
import json
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from datetime import datetime
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # フォールバック（JST日付が不要ならNoneでも動作）

__all__ = [
    "Area", "LogData", "load",
    "_detect_file_type", "_parse_vrchat_txt", "_parse_yaiba_csv", "_parse_yaiba_json",
    "_apply_anonymize", "_apply_time_sync", "_resample_positions", "_finalize_frames"
]

# =========================================================
# プロジェクトルート/出力ルートの決定（src/engine/core からの相対）
# =========================================================
SRC_DIR = Path(__file__).resolve().parents[2]   # .../<PROJECT_ROOT>/src
PROJECT_ROOT = SRC_DIR.parent                   # .../<PROJECT_ROOT>
DEFAULT_DATA_ROOT = PROJECT_ROOT / "YAIBA_data" # 成果物は常にプロジェクト直下の YAIBA_data へ

# ----------------------------
# ロガー
# ----------------------------
def _build_logger() -> logging.Logger:
    logger = logging.getLogger("YAIBA_LOADER")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
    return logger

LOGGER = _build_logger()

# ----------------------------
# ユーティリティ
# ----------------------------
def _to_datetime_utc(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y.%m.%d %H:%M:%S")
    except Exception:
        return None

def _event_day_jst(dt_utc: pd.Series) -> pd.Series:
    if ZoneInfo is None:
        return dt_utc.dt.date
    jst = ZoneInfo("Asia/Tokyo")
    return (dt_utc.dt.tz_localize("UTC").dt.tz_convert(jst)).dt.date

def _hash_name(name: str) -> str:
    import hashlib
    h = hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"U{int(h[:6], 16) % 100000:05d}"

def _ensure_float(s) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return np.nan

def _ensure_int(s) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


# ----------------------------
# Area クラス
# ----------------------------
class Area:
    def __init__(self, position: pd.DataFrame) -> None:
        if position is None or position.empty:
            self.__x_min = self.__x_max = self.__y_min = self.__y_max = self.__z_min = self.__z_max = np.nan
            return
        self.__x_min = float(position["location_x"].min())
        self.__x_max = float(position["location_x"].max())
        self.__y_min = float(position["location_y"].min())
        self.__y_max = float(position["location_y"].max())
        self.__z_min = float(position["location_z"].min())
        self.__z_max = float(position["location_z"].max())

    @property
    def x_min(self) -> float: return self.__x_min
    @property
    def x_max(self) -> float: return self.__x_max
    @property
    def y_min(self) -> float: return self.__y_min
    @property
    def y_max(self) -> float: return self.__y_max
    @property
    def z_min(self) -> float: return self.__z_min
    @property
    def z_max(self) -> float: return self.__z_max


# ----------------------------
# LogData クラス
# ----------------------------
class LogData:
    def __init__(
        self,
        log_file: str,
        span: int,
        is_pseudo: bool,
        time_sync: Optional[datetime],
        position: pd.DataFrame,
        attendance: Optional[pd.DataFrame]
    ) -> None:
        self.__log_file = Path(log_file).resolve()
        self.__span = span
        self.__is_pseudo = is_pseudo
        self.__time_sync = time_sync

        self.__position: pd.DataFrame = position.copy() if position is not None else pd.DataFrame()
        self.__attendance: Optional[pd.DataFrame] = attendance.copy() if attendance is not None else None
        self.__area: Area = Area(self.__position)

        self.__root = DEFAULT_DATA_ROOT
        self.__input_dir = self.__root / "input"
        self.__output_dir = self.__root / "output"
        self.__results_dir = self.__output_dir / "results"
        self.__meta_dir = self.__output_dir / "meta"
        self.__configs_dir = self.__meta_dir / "configs"
        self.__logs_dir = self.__meta_dir / "logs"
        for d in [self.__input_dir, self.__results_dir, self.__configs_dir, self.__logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def position(self) -> pd.DataFrame: return self.__position
    @property
    def attendance(self) -> Optional[pd.DataFrame]: return self.__attendance
    @property
    def area(self) -> Area: return self.__area
    @property
    def span(self) -> int: return self.__span
    @property
    def is_pseudo(self) -> bool: return self.__is_pseudo
    @property
    def time_sync(self) -> Optional[datetime]: return self.__time_sync

    def save(self, filename: str, ver: str, data: Any) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_ver = f"-{ver}" if ver else ""
        out = self.__results_dir / f"{filename}-{ts}{safe_ver}"
        if isinstance(data, pd.DataFrame):
            out = out.with_suffix(".csv")
            data.to_csv(out, index=False)
        elif isinstance(data, (dict, list)):
            out = out.with_suffix(".yaml")
            with open(out, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        else:
            out = out.with_suffix(".txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(str(data))
        LOGGER.info(f"Saved: {out}")

    def save_default_outputs(self, ver: str = "") -> None:
        """
        成果物（実行パラメータYAML / ログ）を出力
        ※ 可視化成果物は出力しない
        """
        params = {
            "log_file": str(self.__log_file),
            "span": self.__span,
            "is_pseudo": self.__is_pseudo,
            "time_sync": None if self.__time_sync is None else self.__time_sync.isoformat(),
        }
        self.__configs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.__configs_dir / f"run_{(ver or 'v1')}_params.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(params, f, allow_unicode=True, sort_keys=False)

        self.__logs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.__logs_dir / f"run{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", "w", encoding="utf-8") as f:
            f.write("YAIBA Loader executed.\n")

    def export(self, zip_name: str = "YAIBA_Visualizer_output.zip") -> None:
        zip_path = self.__root / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in self.__output_dir.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(self.__root))
        LOGGER.info(f"Exported: {zip_path}")


# ----------------------------
# ファイルタイプ判定
# ----------------------------
def _detect_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".csv"]:
        return "yaiba_csv"
    if ext in [".json", ".jsonl"]:
        return "yaiba_json"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(4000)
    if "[Player Position]" in head or "OnPlayerLeft" in head or "OnPlayerJoinComplete" in head:
        return "vrchat_txt"
    try:
        j = json.loads(head.splitlines()[0])
        if isinstance(j, dict) and "type_id" in j:
            return "yaiba_json"
    except Exception:
        pass
    if "timestamp,player_id,player_name" in head:
        return "yaiba_csv"
    return "vrchat_txt"

# ----------------------------
# VRChat 生ログパーサ
# ----------------------------
_JOIN_LEFT_RE = re.compile(
    r"^(?P<dt>\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}).*?\[Behaviour\]\s+(?P<action>OnPlayerJoinComplete|OnPlayerLeft)\s+(?P<name>.+?)(?:\s+\(|$)"
)
_POS_RE = re.compile(
    r"^(?P<dt>\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}).*?\[Player Position](?P<csv>.+)$"
)

def _parse_vrchat_txt(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_pos: List[dict] = []
    rows_att: List[dict] = []
    name_to_uid: Dict[str, int] = {}
    next_uid = 1

    def _uid(name: str) -> int:
        nonlocal next_uid
        if name not in name_to_uid:
            name_to_uid[name] = next_uid
            next_uid += 1
        return name_to_uid[name]

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _JOIN_LEFT_RE.match(line)
            if m:
                dt = _to_datetime_utc(m.group("dt"))
                action_raw = m.group("action")
                name = m.group("name").strip().strip('"')
                if dt is not None and name:
                    rows_att.append({
                        "second": dt,
                        "action": "join" if action_raw == "OnPlayerJoinComplete" else "left",
                        "user_id": _uid(name),
                        "user_name": name,
                        "is_error": False
                    })
                else:
                    rows_att.append({
                        "second": None,
                        "action": None,
                        "user_id": None,
                        "user_name": name or "",
                        "is_error": True
                    })
                continue

            m2 = _POS_RE.match(line)
            if m2:
                dt = _to_datetime_utc(m2.group("dt"))
                csv_payload = m2.group("csv")
                try:
                    for row in csv.reader([csv_payload]):
                        row = [c.strip() for c in row]
                        pid = _ensure_int(row[0]) if len(row) > 0 else None
                        name = row[1].strip('"') if len(row) > 1 else ""
                        x = _ensure_float(row[2]) if len(row) > 2 else np.nan
                        y = _ensure_float(row[3]) if len(row) > 3 else np.nan
                        z = _ensure_float(row[4]) if len(row) > 4 else np.nan
                        r1 = _ensure_float(row[5]) if len(row) > 5 else np.nan
                        r2 = _ensure_float(row[6]) if len(row) > 6 else np.nan
                        r3 = _ensure_float(row[7]) if len(row) > 7 else np.nan
                        dx = _ensure_float(row[8]) if len(row) > 8 else np.nan
                        dy = _ensure_float(row[9]) if len(row) > 9 else np.nan
                        dz = _ensure_float(row[10]) if len(row) > 10 else np.nan
                        isvr = str(row[11]).lower() == "true" if len(row) > 11 else None
                        uid = pid if pid is not None else _uid(name)
                        rows_pos.append({
                            "second": dt,
                            "user_id": uid,
                            "user_name": name,
                            "location_x": x, "location_y": y, "location_z": z,
                            "rotation_1": r1, "rotation_2": r2, "rotation_3": r3,
                            "location_dx": dx, "location_dy": dy, "location_dz": dz,
                            "is_vr": isvr,
                            "is_error": False
                        })
                except Exception:
                    rows_pos.append({
                        "second": None, "user_id": None, "user_name": "",
                        "location_x": np.nan, "location_y": np.nan, "location_z": np.nan,
                        "rotation_1": np.nan, "rotation_2": np.nan, "rotation_3": np.nan,
                        "location_dx": np.nan, "location_dy": np.nan, "location_dz": np.nan,
                        "is_vr": None, "is_error": True
                    })

    pos = pd.DataFrame(rows_pos)
    att = pd.DataFrame(rows_att) if rows_att else pd.DataFrame(columns=["second","action","user_id","user_name","is_error"])
    return pos, att


# ----------------------------
# YAIBA CSV / JSON パーサ
# ----------------------------
def _parse_yaiba_csv(path: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = pd.read_csv(path)
    rename = {"timestamp": "second", "player_id": "user_id", "player_name": "user_name"}
    df = df.rename(columns=rename)
    if not np.issubdtype(df["second"].dtype, np.datetime64):
        df["second"] = pd.to_datetime(df["second"], errors="coerce")
    for c in ["location_x","location_y","location_z","rotation_1","rotation_2","rotation_3",
              "location_dx","location_dy","location_dz"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_vr" in df.columns:
        df["is_vr"] = df["is_vr"].astype("boolean")
    df["is_error"] = df.isna().any(axis=1)
    attendance = None
    return df, attendance

def _parse_yaiba_json(path: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    rows_pos, rows_att = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type_id")
            ts = obj.get("timestamp")
            dt = pd.to_datetime(ts, errors="coerce")
            if t == "yaiba/player_position":
                rows_pos.append({
                    "second": dt.to_pydatetime() if not pd.isna(dt) else None,
                    "user_id": obj.get("player_id"),
                    "user_name": obj.get("player_name"),
                    "location_x": obj.get("location_x"),
                    "location_y": obj.get("location_y"),
                    "location_z": obj.get("location_z"),
                    "rotation_1": obj.get("rotation_1"),
                    "rotation_2": obj.get("rotation_2"),
                    "rotation_3": obj.get("rotation_3"),
                    "location_dx": obj.get("location_dx"),
                    "location_dy": obj.get("location_dy"),
                    "location_dz": obj.get("location_dz"),
                    "is_vr": obj.get("is_vr"),
                    "is_error": False
                })
            elif t in ("yaiba/player_join", "yaiba/player_left"):
                rows_att.append({
                    "second": dt.to_pydatetime() if not pd.isna(dt) else None,
                    "action": "join" if t.endswith("join") else "left",
                    "user_id": obj.get("player_id"),
                    "user_name": obj.get("player_name"),
                    "is_error": False
                })
    pos = pd.DataFrame(rows_pos)
    att = pd.DataFrame(rows_att) if rows_att else None
    return pos, att

# ----------------------------
# 加工（匿名化、time_sync、リサンプリング等）
# ----------------------------
def _apply_anonymize(df: pd.DataFrame, is_pseudo: bool) -> pd.DataFrame:
    if not is_pseudo or df is None or df.empty or "user_name" not in df.columns:
        return df
    map_cache: Dict[str, str] = {}
    def _map(n):
        if pd.isna(n) or n == "":
            return n
        if n not in map_cache:
            map_cache[n] = _hash_name(str(n))
        return map_cache[n]
    df = df.copy()
    df["user_name"] = df["user_name"].astype(str).map(_map)
    return df

def _apply_time_sync(df: Optional[pd.DataFrame], time_sync: Optional[datetime]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or time_sync is None:
        return df
    df = df.copy()
    if "second" not in df.columns:
        return df
    first = df["second"].dropna().min()
    if pd.isna(first):
        return df
    delta = time_sync - first
    df["second"] = df["second"] + delta
    return df

def _resample_positions(df: pd.DataFrame, span: int) -> pd.DataFrame:
    if df is None or df.empty or span <= 1:
        return df
    req_cols = ["second","user_id","location_x","location_y","location_z",
                "rotation_1","rotation_2","rotation_3"]
    for c in req_cols:
        if c not in df.columns:
            df[c] = np.nan

    g = df.copy()
    g["second"] = pd.to_datetime(g["second"])
    g = g.dropna(subset=["second","user_id"])
    g["second"] = g["second"].dt.floor(f"{span}S")
    g = g.sort_values(["user_id","second"])
    agg_last = {
        "location_x":"last","location_y":"last","location_z":"last",
        "rotation_1":"last","rotation_2":"last","rotation_3":"last",
        "user_name":"last","is_vr":"last","is_error":"max"
    }
    g2 = g.groupby(["user_id","second"], as_index=False).agg(agg_last)
    g2 = g2.sort_values(["user_id","second"])
    for ax in ["x","y","z"]:
        g2[f"location_d{ax}"] = g2.groupby("user_id")[f"location_{ax}"].diff()
    g2["event_day"] = _event_day_jst(pd.to_datetime(g2["second"]))
    return g2


def _finalize_frames(pos: pd.DataFrame, att: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if pos is None or pos.empty:
        pos = pd.DataFrame(columns=[
            "second","user_id","user_name",
            "location_x","location_y","location_z",
            "rotation_1","rotation_2","rotation_3",
            "location_dx","location_dy","location_dz",
            "is_vr","event_day","is_error"
        ])
    else:
        pos["second"] = pd.to_datetime(pos["second"], errors="coerce")
        if "user_id" in pos.columns:
            pos["user_id"] = pd.to_numeric(pos["user_id"], errors="coerce").astype("Int64")
        for c in ["location_x","location_y","location_z",
                  "rotation_1","rotation_2","rotation_3",
                  "location_dx","location_dy","location_dz"]:
            if c in pos.columns:
                pos[c] = pd.to_numeric(pos[c], errors="coerce")
        if "is_vr" in pos.columns:
            pos["is_vr"] = pos["is_vr"].astype("boolean")
        if "event_day" not in pos.columns:
            pos["event_day"] = _event_day_jst(pos["second"])
        if "is_error" not in pos.columns:
            pos["is_error"] = pos.isna().any(axis=1)
        cols = ["second","user_id","user_name",
                "location_x","location_y","location_z",
                "rotation_1","rotation_2","rotation_3",
                "location_dx","location_dy","location_dz",
                "is_vr","event_day","is_error"]
        pos = pos.reindex(columns=cols)

    if att is not None and not att.empty:
        att = att.copy()
        att["second"] = pd.to_datetime(att["second"], errors="coerce")
        if "user_id" in att.columns:
            att["user_id"] = pd.to_numeric(att["user_id"], errors="coerce").astype("Int64")
        if "is_error" not in att.columns:
            att["is_error"] = att.isna().any(axis=1)
        att = att.reindex(columns=["second","action","user_id","user_name","is_error"])
    else:
        att = None

    return pos, att


# ----------------------------
# 公開 API: load
# ----------------------------
def load(
    log_file: str,
    span: int = 1,
    is_pseudo: bool = True,
    time_sync: Optional[datetime] = None,
) -> LogData:
    """
    設計書仕様の load
      - ローカル入力ファイルは削除しない
      - YAIBA_data/input 配下はクリーンアップして空にする
    """
    p = Path(log_file)
    if not p.exists():
        raise ValueError("[2101] 指定したログファイルが見つかりません: {}".format(log_file))
    if not isinstance(span, int) or not (1 <= span <= 3600):
        raise ValueError("[2104] span は 1〜3600 の int で指定してください")
    if not isinstance(is_pseudo, bool):
        raise ValueError("[2104] is_pseudo は bool で指定してください")
    if time_sync is not None and not isinstance(time_sync, datetime):
        raise ValueError("[2104] time_sync は datetime または None で指定してください")

    try:
        ftype = _detect_file_type(p)
        if ftype == "vrchat_txt":
            pos, att = _parse_vrchat_txt(p)
        elif ftype == "yaiba_csv":
            pos, att = _parse_yaiba_csv(p)
        elif ftype == "yaiba_json":
            pos, att = _parse_yaiba_json(p)
        else:
            pos, att = _parse_vrchat_txt(p)

        pos = _apply_anonymize(pos, is_pseudo)
        if att is not None:
            att = _apply_anonymize(att, is_pseudo)
        pos = _apply_time_sync(pos, time_sync)
        if att is not None:
            att = _apply_time_sync(att, time_sync)
        pos = _resample_positions(pos, span)
        pos, att = _finalize_frames(pos, att)

        logdata = LogData(
            log_file=log_file,
            span=span,
            is_pseudo=is_pseudo,
            time_sync=time_sync,
            position=pos,
            attendance=att
        )

        # ====== クリーンアップ（inputのみ）======
        try:
            input_dir = DEFAULT_DATA_ROOT / "input"
            if input_dir.exists():
                for _child in input_dir.iterdir():
                    try:
                        if _child.is_file() or _child.is_symlink():
                            _child.unlink()
                        elif _child.is_dir():
                            shutil.rmtree(_child)
                    except Exception as e:
                        LOGGER.warning(f"[2102] input残骸の削除に失敗: {_child} ({e})")
        except Exception as e:
            LOGGER.warning(f"[2102] inputディレクトリのクリーンアップに失敗: {e}")

        return logdata
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"[2100] ログ解釈処理で致命的エラーが発生しました: {e}") from e


# ----------------------------
# 補助: ファイル名ベース
# ----------------------------
def _build_name_base(input_name: str, df_pos: pd.DataFrame) -> str:
    fname = Path(input_name).stem
    if df_pos is None or df_pos.empty or "event_day" not in df_pos.columns:
        day = "unknown"
    else:
        day = str(df_pos["event_day"].dropna().iloc[0]) if not df_pos["event_day"].dropna().empty else "unknown"
    nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{day}-{fname}-{nowstr}"


# ----------------------------
# サービス実行エントリポイント
# ----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="YAIBA Loader Service Runner")
    ap.add_argument("--user-path", required=True, help="入力ログファイルのパス")
    ap.add_argument("--span", type=int, default=1, help="リサンプリング間隔（秒, default=1）")
    ap.add_argument("--is-pseudo", action="store_true", default=True, help="匿名化を有効にする（default=True）")
    ap.add_argument("--no-is-pseudo", dest="is_pseudo", action="store_false", help="匿名化を無効にする")
    ap.add_argument("--time-sync", type=str, default=None,
                    help="時間同期基準 (例: '2025-09-06 22:55:27')")
    args = ap.parse_args()

    # time_sync の変換
    time_sync_dt = None
    if args.time_sync:
        try:
            time_sync_dt = datetime.strptime(args.time_sync, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            LOGGER.warning("[WARN] time-sync の形式が不正です。無視します。")

    # YAIBA ローダ呼び出し
    ld = load(
        log_file=args.user_path,
        span=args.span,
        is_pseudo=args.is_pseudo,
        time_sync=time_sync_dt,
    )

    # === 開発・実装担当向け確認出力 ===
    print("\n=== Position (head) ===")
    print(ld.position.head())
    print(ld.position.info())

    if ld.attendance is not None and not ld.attendance.empty:
        print("\n=== Attendance (head) ===")
        print(ld.attendance.head())
        print(ld.attendance.info())
    else:
        print("\n[INFO] attendance は None/empty です")

