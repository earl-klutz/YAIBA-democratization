import os
from pathlib import Path

# ルートは環境変数で差し替え可（Windows/Colab両対応）
DEFAULT_RESULT_ROOT = str((Path.cwd() / "YAIBA_data/output").resolve())
RESULT_ROOT = os.getenv("YAIBA_RESULT_ROOT") or DEFAULT_RESULT_ROOT
META_ROOT = os.getenv("YAIBA_META_ROOT") or str(Path(RESULT_ROOT) / "meta")

def build_basename(event_day: str, filename: str, dt: str, ver: str, *, duration: int | None = None) -> str:
    # event_day – filename – datetime – (movieのみ)duration – _ver
    core = f"{event_day}-{filename}-{dt}"
    if duration is not None:
        core += f"-{duration}s"
    return f"{core}_{ver}"

def result_path(kind: str, basename: str) -> str:
    """
    出力ファイルの絶対パスを生成。
    優先: YAIBA_RESULT_ROOT > RESULT_ROOT
    """
    root = Path(os.getenv("YAIBA_RESULT_ROOT", RESULT_ROOT))

    subdir_map  = {"movie": "movies", "image": "images", "table": "tables"}
    ext_map     = {"movie": "mp4",    "image": "png",    "table": "csv"}
    prefix_map  = {"movie": "movie_xz","image":"artifact","table":"artifact"}

    subdir = subdir_map.get(kind, "")
    ext    = ext_map.get(kind, "dat")
    prefix = prefix_map.get(kind, "artifact")

    out_dir = root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    return str(out_dir / f"{prefix}-{basename}.{ext}")

def meta_paths(dt: str, ver: str) -> dict:
    return {
        "config_path": os.path.join(META_ROOT, "configs", f"run_{ver}_params.yaml"),
        "log_path":    os.path.join(META_ROOT, "logs",    f"run_{dt}.log"),
    }
