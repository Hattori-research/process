"""
config.py

config.toml の読み込み・書き換えを行う共通モジュール

使い方:
    import config
    cfg = config.load()                       # dict で取得（1回目のみファイルを読む）
    cfg["motor"]["port"]
    config.path("stereo_params")              # [paths] の値を process/ 基準の絶対パスで取得
    config.update("marker", {"hsv_lower": [80, 10, 120]})   # 値の書き換え（コメント・書式は保持）
"""

import os
import re
import json
import tomllib

ROOT        = os.path.dirname(os.path.abspath(__file__))   # process/
CONFIG_PATH = os.path.join(ROOT, "config.toml")

_cache = None


def load(reload=False):
    """config.toml を読み込んで dict を返す（キャッシュあり）"""
    global _cache
    if _cache is None or reload:
        with open(CONFIG_PATH, "rb") as f:
            _cache = tomllib.load(f)
    return _cache


def path(key):
    """[paths] の値を process/ 基準の絶対パスに変換して返す"""
    return os.path.join(ROOT, load()["paths"][key])


# ========================================================
# 書き換え
# ========================================================
def _format_value(v):
    """Python の値を TOML の値表記に変換"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if hasattr(v, "tolist"):                 # numpy 配列・numpy スカラ
        return _format_value(v.tolist())
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)                       # 16.0 → "16.0"（整数化しない）
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_format_value(x) for x in v) + "]"
    raise TypeError(f"config.toml に書き込めない型です: {type(v)}")


def _split_comment(text):
    """'値  # コメント' を ('値', '  # コメント') に分ける（文字列内の # は無視）"""
    in_str = None
    for i, ch in enumerate(text):
        if in_str:
            if ch == "\\" and in_str == '"':
                continue
            if ch == in_str and (i == 0 or text[i - 1] != "\\"):
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == "#":
            j = i
            while j > 0 and text[j - 1] in " \t":
                j -= 1
            return text[:j], text[j:]
    return text.rstrip(), text[len(text.rstrip()):]


_SECTION_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
_KEY_RE     = re.compile(r"^(\s*)([A-Za-z0-9_\-]+)(\s*=\s*)(.*)$")


def update(section, values):
    """
    [section] 内の key = 値 を書き換える。コメント・空行・並び順は保持する。
    存在しない key はセクション末尾に、存在しないセクションはファイル末尾に追加する。
    書き込み前に TOML として正しいか検証し、途中で失敗してもファイルは壊れない。
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    remaining = dict(values)
    cur_section = None
    last_key_idx = None      # 対象セクション内で最後に key があった行

    for i, line in enumerate(lines):
        m_sec = _SECTION_RE.match(line)
        if m_sec:
            cur_section = m_sec.group(1).strip()
            continue
        if cur_section != section:
            continue
        m_key = _KEY_RE.match(line)
        if not m_key:
            continue
        last_key_idx = i
        key = m_key.group(2)
        if key in remaining:
            indent, _, eq, rest = m_key.groups()
            _, comment = _split_comment(rest)
            lines[i] = f"{indent}{key}{eq}{_format_value(remaining.pop(key))}{comment}"

    if remaining:
        new_lines = [f"{k} = {_format_value(v)}" for k, v in remaining.items()]
        if last_key_idx is not None:
            lines[last_key_idx + 1:last_key_idx + 1] = new_lines
        else:
            while lines and lines[-1] == "":
                lines.pop()
            lines += ["", f"[{section}]"] + new_lines + [""]

    text = "\n".join(lines)
    tomllib.loads(text)      # 検証（失敗したら例外で中断し、ファイルは変更しない）

    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp_path, CONFIG_PATH)

    load(reload=True)
    print(f"[config] [{section}] を更新: {', '.join(values.keys())}")
