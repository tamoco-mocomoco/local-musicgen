#!/usr/bin/env python3
"""Claude Code / Claude Desktop から呼べる MCP サーバー (stdio)。音楽生成サーバーの HTTP API を薄く包む。

先に python3 run.py でサーバーを起動しておく。mcp パッケージが必要 (uv run --with mcp で動かすのが手軽):
  claude mcp add music -- uv run --no-project --with "mcp>=1.2" python /ABS/PATH/local-music/mcp_server.py
環境変数: MUSIC_URL (既定 http://localhost:8100)、MUSIC_OUTPUT_DIR (既定 このフォルダの outputs/)
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore

MUSIC_URL = os.environ.get("MUSIC_URL", "http://localhost:8100").rstrip("/")
OUTPUT_DIR = Path(os.environ.get("MUSIC_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs"))

STATIC_INSTRUCTIONS = (
    "ローカルの音楽生成サーバー (ACE-Step 1.5)。generate_music で BGM や歌入りの曲を作って WAV で保存する。"
    "caption は英語で曲調・楽器・テンポ感・用途を書く (例: 'cute cheerful chiptune for a kids game title screen, instrumental')。"
    "ゲームの BGM には loop=True が便利。初回はモデルの読み込みに 30〜60 秒かかり、使わなければ数分で自動解放される。"
    "メモリが 24GB 程度のマシンでは、画像生成サーバーのモデルを解放してから使うこと。"
)


def _call(path: str, body: Optional[dict] = None, timeout: float = 1800) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(MUSIC_URL + path, data=data, method="GET" if body is None else "POST",
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:
            pass
        raise ToolError(f"音楽生成サーバーのエラー (HTTP {e.code}): {detail}")
    except (urllib.error.URLError, ConnectionError) as e:
        raise ToolError(f"音楽生成サーバー {MUSIC_URL} に接続できません ({e})。local-music で python3 run.py を実行してください")


def _instructions() -> str:
    try:
        h = _call("/health", timeout=5)
    except Exception:
        return STATIC_INSTRUCTIONS
    st = h.get("settings", {})
    fp = (h.get("footprint_gb") or {}).get("now")
    return STATIC_INSTRUCTIONS + (
        f" 現在の設定: 言語モデル {st.get('lm_model')}、省メモリ {'有効' if st.get('lean') else '無効'}、"
        f"自動解放 {st.get('idle_unload_seconds')} 秒。モデルは{'読み込み済み' if h.get('loaded') else '未読み込み'}"
        + (f" (サーバーのメモリ {fp}GB)" if fp is not None else "") + "。"
    )


server = _Server("music", instructions=_instructions())


@server.tool()
def server_status() -> dict:
    """音楽生成サーバーの状態 (モデルを読み込み済みか、メモリ、設定)。"""
    return _call("/health", timeout=10)


@server.tool()
def load_model() -> dict:
    """モデルを今すぐ読み込む (30〜60 秒)。続けて何曲も作るときの待ち時間を前倒しできる。"""
    return _call("/load", {})


@server.tool()
def unload_model() -> dict:
    """モデルを解放してメモリを空ける。画像生成など別の重い処理の前に使う。"""
    return _call("/unload", {})


@server.tool()
def generate_music(
    caption: str,
    duration: float = 30.0,
    instrumental: bool = True,
    lyrics: Optional[str] = None,
    vocal_language: Optional[str] = None,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    seed: Optional[int] = None,
    loop: bool = False,
    use_lm: Optional[bool] = None,
    out_path: Optional[str] = None,
) -> str:
    """曲を生成して WAV を保存し、保存先とメタ情報を返す。

    - caption: 英語で曲調・楽器・テンポ感・用途 (例: 'calm felt piano and music box lullaby, slow, instrumental')
    - duration: 秒 (5〜240)。言語モデル使用時は曲が短めに終わることがあり、末尾の無音は自動で切る
    - instrumental=False で歌入り。lyrics に [Verse] [Chorus] などのタグ付き歌詞、vocal_language に en / ja など
    - loop=True: 末尾のフェードと無音を切り、終端を始端にクロスフェードしてゲームでループ再生しやすくする
    - use_lm=False: 言語モデルを使わない (指定の長さを最後まで使い切りやすい)
    - out_path: 保存先。省略時は outputs/ にサーバー命名で保存
    """
    body: dict = dict(caption=caption, duration=duration, instrumental=instrumental, loop=loop, return_base64=True)
    for k, v in (("lyrics", lyrics), ("vocal_language", vocal_language), ("bpm", bpm), ("keyscale", keyscale),
                 ("seed", seed), ("use_lm", use_lm)):
        if v is not None:
            body[k] = v
    res = _call("/generate", body)
    data = base64.b64decode(res.pop("base64"))
    if out_path:
        path = Path(out_path).expanduser()
        if not path.is_absolute():
            path = OUTPUT_DIR / path
    else:
        path = OUTPUT_DIR / res["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    info = res.get("loop") or res.get("trim") or {}
    length = info.get("loop_seconds") or info.get("content_seconds")
    return (f"saved: {path}\n  seed={res['seed']} length={length}s generate={res['seconds']}s lm={res.get('lm')}"
            + (f"\n  loop: crossfade={info.get('crossfade_seconds')}s, cut head={info.get('trimmed_head_seconds')}s "
               f"tail={info.get('trimmed_tail_seconds')}s" if res.get("loop") else ""))


if __name__ == "__main__":
    server.run("stdio")
