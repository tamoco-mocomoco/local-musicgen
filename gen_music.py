#!/usr/bin/env python3
"""依存ゼロの CLI。音楽生成サーバーに投げて WAV を保存する。

  python3 gen_music.py "cute cheerful chiptune for a kids game title screen, instrumental" -o title.wav
  python3 gen_music.py "calm piano lullaby, instrumental" --duration 45 --loop -o nap.wav
  python3 gen_music.py "upbeat kids pop song" --vocal en --lyrics lyrics.txt -o song.wav
  python3 gen_music.py --status | --load | --unload
  MUSIC_URL=http://other-host:8100 python3 gen_music.py "..."

保存したファイルのパスを stdout に 1 行出す。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request


def call(url: str, path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method="GET" if body is None else "POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate music via the local music server")
    p.add_argument("caption", nargs="?", help="曲の説明 (英語推奨)")
    p.add_argument("-o", "--out", help="出力ファイル名 (省略時は outputs/ にサーバー命名で保存)")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--bpm", type=int)
    p.add_argument("--key", help="例: C Major")
    p.add_argument("--seed", type=int)
    p.add_argument("--loop", action="store_true", help="ループしやすく加工する")
    p.add_argument("--crossfade", type=float, default=1.5, help="ループのつなぎ目のクロスフェード秒数")
    p.add_argument("--no-lm", action="store_true", help="言語モデルを使わない (指定の長さを最後まで使い切りやすい)")
    p.add_argument("--no-trim", action="store_true", help="末尾の無音を切らない")
    p.add_argument("--vocal", metavar="LANG", help="歌入りにする (en, ja など)。--lyrics と併用")
    p.add_argument("--lyrics", help="歌詞ファイルのパス、または歌詞の文字列")
    p.add_argument("--status", action="store_true")
    p.add_argument("--load", action="store_true")
    p.add_argument("--unload", action="store_true")
    p.add_argument("--url", default=os.environ.get("MUSIC_URL", "http://localhost:8100"))
    a = p.parse_args()

    try:
        if a.status or a.load or a.unload:
            path = "/health" if a.status else ("/load" if a.load else "/unload")
            res = call(a.url, path, None if a.status else {})
            res.pop("settings", None) if not a.status else None
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0
        if not a.caption:
            p.error("caption が必要です")
        body = dict(caption=a.caption, duration=a.duration, loop=a.loop, loop_crossfade=a.crossfade,
                    trim=not a.no_trim, return_base64=True, instrumental=not a.vocal)
        if a.no_lm:
            body["use_lm"] = False
        if a.bpm:
            body["bpm"] = a.bpm
        if a.key:
            body["keyscale"] = a.key
        if a.seed is not None:
            body["seed"] = a.seed
        if a.vocal:
            body["vocal_language"] = a.vocal
            lyr = a.lyrics or ""
            body["lyrics"] = open(lyr).read() if lyr and os.path.isfile(lyr) else lyr
        res = call(a.url, "/generate", body)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"cannot reach {a.url} ({e}); is the server running? (python3 run.py)", file=sys.stderr)
        return 2

    out = a.out or os.path.join("outputs", res["file"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(base64.b64decode(res["base64"]))
    print(out)
    info = res.get("loop") or res.get("trim") or {}
    length = info.get("loop_seconds") or info.get("content_seconds")
    print(f"  seed={res['seed']} {res['seconds']}s lm={res.get('lm')} length={length}s"
          + (" loop" if res.get("loop") else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
