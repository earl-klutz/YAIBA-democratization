# -*- coding: utf-8 -*-
"""
fontconfig.py
YAIBA-BI プロジェクト共通のフォント設定モジュール。
Windows・Colab・Linux の環境差を吸収し、日本語ラベルを確実に表示可能にする。
"""
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_fonts() -> None:
    return

    """環境依存フォント設定を適用"""
    """
    次期以降の開発で要検討
    if platform.system() == "Windows":
        # --- Windows: メイリオを使用 ---
        plt.rcParams["font.family"] = "Meiryo"
    else:
        # --- Linux/Colab ---
        colab_font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
        if os.path.exists(colab_font_path):
            font_prop = fm.FontProperties(fname=colab_font_path)
            plt.rcParams["font.family"] = font_prop.get_name()
        else:
            # フォールバック候補
            plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    """
