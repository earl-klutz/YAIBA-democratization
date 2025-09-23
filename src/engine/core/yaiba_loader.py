# vrc_to_std_outputs.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

import yaiba
import pandas as pd

# === 入出力（変更しない） ===
LOG_PATH = Path(r"C:\Users\tinyt\Downloads\output.txt")  # VRChat生ログ
OUT_DIR  = Path(r"C:\Users\tinyt\Videos\my_project\YAIBA\Aproject\YAIBA-democratization\src\engine\core\YAIBA_output_min")

# === 列定義（Must準拠） ===
POS_COLS = [
    "second",
    "user_id", "user_name",
    "location_x", "location_y", "location_z",
    "rotation_1", "rotation_2", "rotation_3",
    "location_dx", "location_dy", "location_dz",
    "is_vr", "event_day", "is_error",
]
ATT_COLS = ["second", "action", "user_id", "user_name", "is_error"]

ALIASES = {
    "second":   ["second", "timestamp", "time", "created_at"],
    "user_id":  ["user_id", "player_id", "userid", "uid"],
    "user_name":["user_name", "player_name", "display_name", "name", "pseudo_user_name", "username"],
    # attendanceの候補
    "actionish":["action", "event", "event_type", "kind"],
    "textish":  ["type", "type_id", "message", "log", "detail"],
}

# ---------- Helpers ----------
def col_or(df: pd.DataFrame, key: str) -> pd.Series | None:
    for c in ALIASES.get(key, []):
        if c in df.columns:
            return df[c]
    return None

def ensure_utc(s: pd.Series | None) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def event_day_jst(second_utc: pd.Series) -> pd.Series:
    try:
        from zoneinfo import ZoneInfo
        jst = ZoneInfo("Asia/Tokyo")
        return pd.to_datetime(second_utc, utc=True).dt.tz_convert(jst).dt.normalize().dt.date
    except Exception:
        return pd.to_datetime(second_utc, utc=True).dt.date

def norm360(s: pd.Series) -> pd.Series:
    return s.map(lambda v: (float(v) % 360.0) if pd.notna(v) else v)

def fill_user_fields(df: pd.DataFrame) -> pd.DataFrame:
    # user_id
    if ("user_id" not in df.columns) or df["user_id"].isna().all():
        alt = col_or(df, "user_id")
        if alt is not None:
            df["user_id"] = pd.to_numeric(alt, errors="coerce").astype("Int64")
        else:
            df["user_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")

    # user_name
    if ("user_name" not in df.columns) or df["user_name"].isna().all():
        alt = col_or(df, "user_name")
        if alt is not None:
            df["user_name"] = alt.astype("string")
        else:
            df["user_name"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df["user_name"] = df["user_name"].astype("string")

    # 名前があるのにIDが全欠損 → 安定採番（1始まり）
    if df["user_id"].isna().all() and df["user_name"].notna().any():
        codes, _ = pd.factorize(df["user_name"], sort=True)
        ids = pd.Series(codes).astype("Int64")
        df["user_id"] = ids.where(ids >= 0, other=pd.NA) + 1

    # 名前欠損のみ U##### 補完（既存は上書きしない）
    na = df["user_name"].isna()
    if na.any():
        df.loc[na, "user_name"] = df.loc[na, "user_id"].map(
            lambda v: f"U{int(v):05d}" if pd.notna(v) else pd.NA
        ).astype("string")
    return df

# ---------- Position ----------
def coerce_position(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=df_raw.index)
    df["second"] = ensure_utc(col_or(df_raw, "second"))

    # ID/名前
    df = pd.concat([df, df_raw], axis=1)
    df = fill_user_fields(df)

    # 数値列
    for c in ["location_x","location_y","location_z",
              "rotation_1","rotation_2","rotation_3",
              "location_dx","location_dy","location_dz"]:
        df[c] = pd.to_numeric(df_raw.get(c), errors="coerce")

    # 角度正規化
    for c in ["rotation_1","rotation_2","rotation_3"]:
        df[c] = norm360(df[c])

    # ΔL補完（3軸とも欠損のとき）
    if df[["location_dx","location_dy","location_dz"]].isna().all().all():
        df = df.sort_values(["user_id","second"])
        for axis in ("x","y","z"):
            df[f"location_d{axis}"] = df[f"location_{axis}"].groupby(df["user_id"], dropna=False).diff().fillna(0.0)

    df["is_vr"] = (df_raw.get("is_vr").astype("boolean")
                   if "is_vr" in df_raw.columns else pd.Series([pd.NA]*len(df), dtype="boolean"))
    df["is_error"] = (df_raw.get("is_error").fillna(False).astype(bool)
                      if "is_error" in df_raw.columns else False)
    df["event_day"] = event_day_jst(df["second"])

    for c in POS_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[POS_COLS]

# ---------- Attendance ----------
JOIN_PATTERNS = ("join", "joined", "onplayerjoin")
LEFT_PATTERNS = ("left", "onplayerleft")

def _detect_explicit_attendance(df_raw: pd.DataFrame) -> pd.DataFrame:
    """join/left の明示イベントがあれば抽出"""
    # まず action系の列に 'join' / 'left' が入っていないか
    act = col_or(df_raw, "actionish")
    txt = col_or(df_raw, "textish")

    # 判定用の文字列列を1本作る
    base = pd.DataFrame(index=df_raw.index)
    if act is not None:
        base["_sig"] = act.astype(str)
    elif txt is not None:
        base["_sig"] = txt.astype(str)
    else:
        # それっぽい列が何も無ければ空
        return pd.DataFrame(columns=ATT_COLS)

    sig = base["_sig"].str.lower()
    mask = (
        sig.str.contains("|".join(JOIN_PATTERNS), na=False) |
        sig.str.contains("|".join(LEFT_PATTERNS), na=False)
    )
    base = df_raw.loc[mask].copy()
    if base.empty:
        return pd.DataFrame(columns=ATT_COLS)

    # second
    base["second"] = ensure_utc(col_or(base, "second"))

    # actionを決める
    if "action" in base.columns:
        base["action"] = base["action"].astype("string")
    else:
        s = (col_or(base, "actionish") if col_or(base, "actionish") is not None else
             (col_or(base, "textish") if col_or(base, "textish") is not None else pd.Series([""]*len(base))))
        s = s.astype(str).str.lower()
        base["action"] = s.map(lambda t:
            "join" if any(k in t for k in JOIN_PATTERNS) else ("left" if any(k in t for k in LEFT_PATTERNS) else pd.NA)
        ).astype("string")

    base = fill_user_fields(base)
    base["is_error"] = False

    for c in ATT_COLS:
        if c not in base.columns:
            base[c] = pd.NA
    return base[ATT_COLS]

def _derive_attendance_from_position(df_pos: pd.DataFrame) -> pd.DataFrame:
    """明示イベントが無い場合のフォールバック：
       各 user_id の最初の出現を join、最後の出現を left として生成"""
    if df_pos.empty:
        return pd.DataFrame(columns=ATT_COLS)

    # user_idが欠損の行は除外（どうしても必要ならuser_nameで採番するが、position側で対応済み）
    g = df_pos.dropna(subset=["user_id"]).sort_values("second").groupby("user_id", dropna=False)

    first = g.first().reset_index()
    last  = g.last().reset_index()

    join_df = pd.DataFrame({
        "second": first["second"],
        "action": pd.Series(["join"] * len(first), dtype="string"),
        "user_id": first["user_id"].astype("Int64"),
        "user_name": first["user_name"].astype("string"),
        "is_error": False,
    })
    left_df = pd.DataFrame({
        "second": last["second"],
        "action": pd.Series(["left"] * len(last), dtype="string"),
        "user_id": last["user_id"].astype("Int64"),
        "user_name": last["user_name"].astype("string"),
        "is_error": False,
    })

    att = pd.concat([join_df, left_df], ignore_index=True).sort_values(["second","user_id"])
    # 列順合わせ
    for c in ATT_COLS:
        if c not in att.columns:
            att[c] = pd.NA
    return att[ATT_COLS]

def extract_attendance(df_raw: pd.DataFrame, df_pos: pd.DataFrame) -> pd.DataFrame:
    """1) 明示イベントを広く検出 → 2) 無ければpositionから推定"""
    att = _detect_explicit_attendance(df_raw)
    if not att.empty:
        return att
    # フォールバック（位置データしか無いログ向け）
    return _derive_attendance_from_position(df_pos)

# ---------- 面積 ----------
def calc_area(df_pos: pd.DataFrame) -> dict:
    return {
        "x_min": float(df_pos["location_x"].min(skipna=True)),
        "x_max": float(df_pos["location_x"].max(skipna=True)),
        "y_min": float(df_pos["location_y"].min(skipna=True)),
        "y_max": float(df_pos["location_y"].max(skipna=True)),
        "z_min": float(df_pos["location_z"].min(skipna=True)),
        "z_max": float(df_pos["location_z"].max(skipna=True)),
    }

# ---------- 実行本体 ----------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) yaibaで生ログ→session_log
    with open(LOG_PATH, encoding="utf-8") as fp:
        session_log = yaiba.parse_vrchat_log(fp)

    # 2) session_log → DataFrame
    df_raw = pd.DataFrame(session_log.log_entries)

    # 3) Must仕様で正規化
    df_pos = coerce_position(df_raw)
    df_att = extract_attendance(df_raw, df_pos)

    # 4) JST時刻でファイル名刻印
    try:
        from zoneinfo import ZoneInfo
        stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    except Exception:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    pos_name = f"position_{stamp}.csv"
    att_name = f"attendance_{stamp}.csv"
    area_name = f"area_{stamp}.json"

    # 5) CSVの second は "YYYY-MM-DD HH:MM:SS"（UTC表記）
    df_pos_out = df_pos.copy()
    df_pos_out["second"] = pd.to_datetime(df_pos_out["second"], utc=True).dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")

    df_att_out = df_att.copy()
    if not df_att_out.empty:
        df_att_out["second"] = pd.to_datetime(df_att_out["second"], utc=True).dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")

    # 6) 保存（I/Oは既存どおり）
    df_pos_out.to_csv(OUT_DIR / pos_name, index=False, encoding="utf-8-sig")
    df_att_out.to_csv(OUT_DIR / att_name, index=False, encoding="utf-8-sig")
    (OUT_DIR / area_name).write_text(json.dumps(calc_area(df_pos), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {OUT_DIR/pos_name}")
    print(f"Wrote: {OUT_DIR/att_name}")
    print(f"Wrote: {OUT_DIR/area_name}")

if __name__ == "__main__":
    main()
