"""ローカル音楽生成サーバー (ACE-Step 1.5)。local-imagegen と同じ作り。

  GET  /health    状態 (読み込み済みか、メモリ、設定)
  POST /generate  曲を生成して outputs/ に保存し、メタ情報を返す
  POST /load      モデルを先に読み込む
  POST /unload    モデルを解放する (使われなければ IDLE_UNLOAD_SECONDS 後に自動でも解放)
  GET  /outputs/<file>
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for noisy in ("httpx", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from server.engine import OUTPUT_DIR, engine  # noqa: E402


class GenerateRequest(BaseModel):
    caption: str = Field(..., min_length=1, description="曲の説明 (英語推奨)。例: cute cheerful chiptune for a kids game title screen, instrumental")
    lyrics: Optional[str] = Field(None, description="歌詞 ([Verse] [Chorus] などのタグ付き)。instrumental=false のときだけ使う")
    instrumental: bool = True
    duration: float = Field(30.0, ge=5, le=240, description="秒")
    bpm: Optional[int] = Field(None, ge=40, le=240)
    keyscale: Optional[str] = Field(None, description="例: C Major, D Minor")
    vocal_language: Optional[str] = Field(None, description="歌入りのときの言語コード (en, ja など)")
    seed: Optional[int] = None
    loop: bool = Field(False, description="末尾のフェードと無音を切り、終端を始端にクロスフェードしてループしやすくする")
    loop_crossfade: float = Field(1.5, ge=0.1, le=8)
    use_lm: Optional[bool] = Field(None, description="false で言語モデルを使わない (指定の長さを最後まで使い切りやすい)")
    trim: bool = Field(True, description="末尾の無音を切る (loop=false のとき)")
    return_base64: bool = False


class Empty(BaseModel):
    pass


app = FastAPI(title="Local Music", version="0.1.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/health")
def health() -> dict:
    s = engine.status()
    return dict(ok=s["error"] is None, **s)


@app.post("/generate")
def generate(req: GenerateRequest) -> JSONResponse:
    try:
        item = engine.generate(
            caption=req.caption, lyrics=req.lyrics, instrumental=req.instrumental, duration=req.duration,
            bpm=req.bpm, keyscale=req.keyscale, vocal_language=req.vocal_language, seed=req.seed,
            loop=req.loop, loop_crossfade=req.loop_crossfade, use_lm=req.use_lm, trim=req.trim)
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    if req.return_base64:
        item["base64"] = base64.b64encode((OUTPUT_DIR / item["file"]).read_bytes()).decode()
    return JSONResponse(item)


@app.post("/load")
def load(_: Optional[Empty] = None) -> dict:
    try:
        engine.load()
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return engine.status()


@app.post("/unload")
def unload(_: Optional[Empty] = None) -> dict:
    unloaded = engine.unload()
    return dict(unloaded=unloaded, **engine.status())
