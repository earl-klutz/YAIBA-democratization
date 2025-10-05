class SpecError(Exception):
    """仕様策定で定義されたエラーコード付き例外。"""

    def __init__(self, code: int, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


__all__ = ["SpecError"]
