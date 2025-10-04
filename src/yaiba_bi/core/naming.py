import os

# ルートは環境変数で差し替え可（Windows/Colab両対応）
RESULT_ROOT = os.getenv("YAIBA_RESULT_ROOT", "/content/YAIBA_data/output/results")
META_ROOT   = os.getenv("YAIBA_META_ROOT",   "/content/YAIBA_data/output/meta")

def build_basename(event_day: str, filename: str, dt: str, ver: str, *, duration: int | None = None) -> str:
    # event_day – filename – datetime – (movieのみ)duration – _ver
    core = f"{event_day}-{filename}-{dt}"
    if duration is not None:
        core += f"-{duration}s"
    return f"{core}_{ver}"

def result_path(kind: str, basename: str) -> str:
    # results直下 movies/images/tables … 設計書準拠
    sub = {"movie": "movies", "image": "images", "table": "tables"}[kind]
    return os.path.join(RESULT_ROOT, sub, f"{'movie_xz' if kind=='movie' else 'artifact'}-{basename}.{ 'mp4' if kind=='movie' else ('png' if kind=='image' else 'csv') }")

def meta_paths(dt: str, ver: str) -> dict:
    return {
        "config_path": os.path.join(META_ROOT, "configs", f"run_{ver}_params.yaml"),
        "log_path":    os.path.join(META_ROOT, "logs",    f"run_{dt}.log"),
    }
