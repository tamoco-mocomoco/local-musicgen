"""生成した曲をループ再生しやすくする後処理。

ACE-Step の曲は終わりに 2〜5 秒のフェードアウトと無音が付くことがあるので、
1. 先頭の無音を切る
2. 末尾のフェードアウトと無音を切る (曲の中央値の音量から fade_db 以上下がった所以降)
3. 終端を始端にクロスフェード (等パワー) して、繰り返したときに継ぎ目が出ないようにする
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _frame_db(mono: np.ndarray, frame: int) -> np.ndarray:
    n = len(mono) // frame
    if n == 0:
        return np.array([-120.0])
    frames = mono[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20 * np.log10(rms + 1e-9)


def make_loop(in_path: Path | str, out_path: Path | str, *, crossfade: float = 1.5,
              fade_db: float = 9.0, silence_db: float = -50.0, frame_seconds: float = 0.05,
              match_window: float = 0.5, match_db: float = 2.0, max_search_seconds: float = 8.0) -> dict:
    x, sr = sf.read(str(in_path), always_2d=True)
    mono = x.mean(axis=1)
    frame = max(1, int(frame_seconds * sr))
    db = _frame_db(mono, frame)

    loud = db > silence_db
    if not loud.any():
        raise ValueError("曲がほぼ無音のためループ化できません")
    start_f = int(np.argmax(loud))
    median_db = float(np.median(db[loud]))
    keep = np.where(db >= median_db - fade_db)[0]
    end_f = int(keep[-1]) + 1 if len(keep) else len(db)

    # 終点は「冒頭と同じくらいの音量」の位置を後ろから探す (フェードアウト途中で切ると継ぎ目で音量が跳ねるため)
    win = max(1, int(match_window / frame_seconds))
    start_db = float(np.mean(db[start_f:start_f + win]))
    search_from = max(start_f + 4 * win, end_f - int(max_search_seconds / frame_seconds))
    chosen = None
    for cand in range(end_f, search_from, -1):
        if abs(float(np.mean(db[cand - win:cand])) - start_db) <= match_db:
            chosen = cand
            break
    if chosen is not None:
        end_f = chosen

    start, end = start_f * frame, min(len(x), end_f * frame)
    body = x[start:end]
    xf = int(min(crossfade * sr, len(body) // 4))
    if xf < 1:
        raise ValueError("曲が短すぎてループ化できません")

    t = np.linspace(0.0, 1.0, xf, endpoint=False)[:, None]
    fade_in, fade_out = np.sin(t * np.pi / 2), np.cos(t * np.pi / 2)
    looped = body[:-xf].copy()
    looped[:xf] = body[:xf] * fade_in + body[-xf:] * fade_out
    peak = float(np.abs(looped).max())
    if peak > 0.999:  # クロスフェードで溢れたときだけ下げる
        looped *= 0.999 / peak
    sf.write(str(out_path), looped, sr, subtype="PCM_16")

    # 継ぎ目の確認用: 終端 -> 始端の 1 サンプル差 (曲中の大きめの差 = 上位 5% との比。1 以下なら段差なし) と、継ぎ目前後 0.5 秒の音量差
    seam = float(np.abs(looped[0] - looped[-1]).mean())
    typical = float(np.percentile(np.abs(np.diff(looped.mean(axis=1))), 95) + 1e-9)
    w = int(match_window * sr)
    def _db(a):
        return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)
    seam_db = float(_db(looped[:w].mean(axis=1)) - _db(looped[-w:].mean(axis=1)))
    return dict(
        original_seconds=round(len(x) / sr, 2), loop_seconds=round(len(looped) / sr, 2),
        trimmed_head_seconds=round(start / sr, 2), trimmed_tail_seconds=round((len(x) - end) / sr, 2),
        crossfade_seconds=round(xf / sr, 2), seam_jump_ratio=round(seam / typical, 2),
        seam_loudness_step_db=round(seam_db, 1), end_matched=chosen is not None,
    )


def trim_silence(path: Path | str, *, silence_db: float = -50.0, keep_seconds: float = 0.3,
                 frame_seconds: float = 0.05) -> dict:
    """末尾の無音を切ってその場で上書きする。言語モデルが短い曲を作ると残りが無音で埋まるため。"""
    x, sr = sf.read(str(path), always_2d=True)
    frame = max(1, int(frame_seconds * sr))
    db = _frame_db(x.mean(axis=1), frame)
    loud = np.where(db > silence_db)[0]
    if not len(loud):
        return dict(original_seconds=round(len(x) / sr, 2), content_seconds=0.0, trimmed_seconds=0.0)
    end = min(len(x), (int(loud[-1]) + 1) * frame + int(keep_seconds * sr))
    trimmed = len(x) - end
    if trimmed > int(0.5 * sr):
        sf.write(str(path), x[:end], sr, subtype="PCM_16")
    else:
        end = len(x)
    return dict(original_seconds=round(len(x) / sr, 2), content_seconds=round(end / sr, 2),
                trimmed_seconds=round((len(x) - end) / sr, 2))
