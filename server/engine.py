"""ACE-Step 1.5 を省メモリ設定で動かす生成エンジン。

- モデルは最初の生成 (または /load) で読み込み、IDLE_UNLOAD_SECONDS 使われなければ自動で解放する
- 生成は 1 件ずつ (ロックで直列化)
- 省メモリ設定 (MUSIC_LEAN=1、既定) は M2 Max での検証結果に基づく:
  PyTorch 側は CPU に置き、MLX へ変換した後に PyTorch 側の DiT decoder と VAE を解放、
  MLX の DiT は bf16 (配布されている重みが bf16 なので劣化なし)、MLX VAE は fp16 で分割 192、
  MLX のキャッシュ上限 1GB、言語モデル 0.6B。常駐約 12GB、生成中ピーク約 15GB。
"""
from __future__ import annotations

import ctypes
import gc
import logging
import os
import random
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("music")

# ---------------------------------------------------------------- settings
LEAN = os.environ.get("MUSIC_LEAN", "1") == "1"

# ACE-Step が import 時に読む環境変数は、import より前に入れておく
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if LEAN:
    os.environ.setdefault("ACESTEP_MLX_VAE_FP16", "1")
    os.environ.setdefault("ACESTEP_MLX_VAE_CHUNK", "192")

ACESTEP_HOME = Path(os.environ.get("ACESTEP_HOME", "")).expanduser()
DIT_CONFIG = os.environ.get("MUSIC_DIT_CONFIG", "acestep-v15-turbo")
LM_MODEL = os.environ.get("MUSIC_LM", "acestep-5Hz-lm-0.6B")  # "none" で言語モデルなし
TORCH_DEVICE = os.environ.get("MUSIC_TORCH_DEVICE", "cpu" if LEAN else "mps")
MLX_DIT_DTYPE = os.environ.get("MUSIC_MLX_DIT_DTYPE", "bf16" if LEAN else "fp32")
MLX_CACHE_GB = os.environ.get("MUSIC_MLX_CACHE_GB", "1" if LEAN else "")
INFERENCE_STEPS = int(os.environ.get("MUSIC_INFERENCE_STEPS", "8"))
IDLE_UNLOAD_SECONDS = int(os.environ.get("IDLE_UNLOAD_SECONDS", "300"))  # 0 で自動解放しない
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parent.parent / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def settings() -> dict:
    return dict(
        lean=LEAN, acestep_home=str(ACESTEP_HOME), dit_config=DIT_CONFIG, lm_model=LM_MODEL,
        torch_device=TORCH_DEVICE, mlx_dit_dtype=MLX_DIT_DTYPE, mlx_cache_gb=MLX_CACHE_GB or None,
        mlx_vae_fp16=os.environ.get("ACESTEP_MLX_VAE_FP16"), mlx_vae_chunk=os.environ.get("ACESTEP_MLX_VAE_CHUNK"),
        inference_steps=INFERENCE_STEPS, idle_unload_seconds=IDLE_UNLOAD_SECONDS, output_dir=str(OUTPUT_DIR),
    )


# ---------------------------------------------------------------- memory (macOS phys_footprint)
class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
        "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
        "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions",
        "ri_cycles", "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time")]


def footprint_gb() -> Optional[dict]:
    """プロセスのメモリ (Activity Monitor の「メモリ」列と同じ値。GPU 側の確保も含む)。macOS 以外は None。"""
    if sys.platform != "darwin":
        return None
    try:
        info = _RusageInfoV4()
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return None
        return dict(now=round(info.ri_phys_footprint / 1e9, 1),
                    lifetime_max=round(info.ri_lifetime_max_phys_footprint / 1e9, 1))
    except Exception:
        return None


# ---------------------------------------------------------------- engine
class MusicEngine:
    def __init__(self) -> None:
        self.dit: Any = None
        self.llm: Any = None
        self.lm_ok = False
        self.lock = threading.Lock()
        self.loading = False
        self.load_error: Optional[str] = None
        self.loaded_in: Optional[float] = None
        self.last_used: Optional[float] = None
        self.unloads = 0
        threading.Thread(target=self._idle_watch, daemon=True).start()

    @property
    def loaded(self) -> bool:
        return self.dit is not None

    # ---- load / unload (self.lock を持って呼ぶ)
    def _load(self) -> None:
        if self.dit is not None:
            return
        if not (ACESTEP_HOME / "acestep").is_dir():
            raise RuntimeError(f"ACESTEP_HOME が ACE-Step 1.5 のチェックアウトを指していません: '{ACESTEP_HOME}'")
        if str(ACESTEP_HOME) not in sys.path:
            sys.path.insert(0, str(ACESTEP_HOME))
        self.loading = True
        t0 = time.time()
        try:
            import mlx.core as mx
            import torch
            from acestep.handler import AceStepHandler
            from acestep.llm_inference import LLMHandler

            if MLX_CACHE_GB:
                mx.set_cache_limit(int(float(MLX_CACHE_GB) * 1e9))

            log.info("loading ACE-Step DiT (%s, torch=%s) ...", DIT_CONFIG, TORCH_DEVICE)
            dit = AceStepHandler()
            status, ok = dit.initialize_service(project_root=str(ACESTEP_HOME), config_path=DIT_CONFIG,
                                                device=TORCH_DEVICE)
            if not ok:
                raise RuntimeError(f"DiT の初期化に失敗: {status}")

            if LEAN:
                # MLX へ変換済みの部品は PyTorch 側の重みを捨てる (形だけ残す)
                if getattr(dit, "use_mlx_dit", False) and getattr(dit, "mlx_decoder", None) is not None:
                    dit.model.decoder.to("meta")
                if getattr(dit, "use_mlx_vae", False) and getattr(dit, "mlx_vae", None) is not None:
                    dit.vae.to("meta")
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            if MLX_DIT_DTYPE in ("bf16", "fp16") and getattr(dit, "mlx_decoder", None) is not None:
                from mlx.utils import tree_map
                target = mx.bfloat16 if MLX_DIT_DTYPE == "bf16" else mx.float16
                dit.mlx_decoder.update(tree_map(
                    lambda v: v.astype(target) if isinstance(v, mx.array) and mx.issubdtype(v.dtype, mx.floating) else v,
                    dit.mlx_decoder.parameters()))
                mx.eval(dit.mlx_decoder.parameters())
                gc.collect()
                mx.clear_cache()

            llm = LLMHandler()
            lm_ok = False
            if LM_MODEL.lower() != "none":
                log.info("loading ACE-Step LM (%s) ...", LM_MODEL)
                lm_status, lm_ok = llm.initialize(checkpoint_dir=str(ACESTEP_HOME / "checkpoints"),
                                                  lm_model_path=LM_MODEL, backend="mlx", device="mps")
                if not lm_ok:
                    log.warning("言語モデルなしで続行します: %s", lm_status)

            self.dit, self.llm, self.lm_ok = dit, llm, lm_ok
            self.loaded_in = round(time.time() - t0, 1)
            self.load_error = None
            log.info("ACE-Step ready in %ss (footprint %s)", self.loaded_in, footprint_gb())
        except Exception as e:
            self.load_error = str(e)
            self._release()
            raise
        finally:
            self.loading = False

    def _release(self) -> None:
        self.dit = None
        self.llm = None
        self.lm_ok = False
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _unload(self, reason: str) -> bool:
        if self.dit is None:
            return False
        before = footprint_gb()
        self._release()
        self.unloads += 1
        log.info("ACE-Step unloaded (%s): footprint %s -> %s", reason, before, footprint_gb())
        return True

    # ---- public
    def load(self) -> None:
        with self.lock:
            self._load()
            self.last_used = time.time()

    def unload(self) -> bool:
        with self.lock:
            return self._unload("requested")

    def _idle_watch(self) -> None:
        while True:
            time.sleep(5)
            if IDLE_UNLOAD_SECONDS <= 0 or self.dit is None or self.last_used is None:
                continue
            if time.time() - self.last_used < IDLE_UNLOAD_SECONDS:
                continue
            if self.lock.acquire(blocking=False):
                try:
                    if self.dit is not None and time.time() - (self.last_used or 0) >= IDLE_UNLOAD_SECONDS:
                        self._unload(f"idle {IDLE_UNLOAD_SECONDS}s")
                finally:
                    self.lock.release()

    def generate(self, *, caption: str, lyrics: Optional[str] = None, instrumental: bool = True,
                 duration: float = 30.0, bpm: Optional[int] = None, keyscale: Optional[str] = None,
                 vocal_language: Optional[str] = None, seed: Optional[int] = None, loop: bool = False,
                 loop_crossfade: float = 1.5, use_lm: Optional[bool] = None, trim: bool = True) -> dict:
        """use_lm: None なら読み込んだ言語モデルを使う。False で言語モデルなし (指定の長さを最後まで使い切りやすい)。
        trim: 末尾の無音を切る (言語モデルが短い曲を作ると残りが無音で埋まるため)。"""
        from server.loop import make_loop, trim_silence

        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        with self.lock:
            self._load()
            from acestep.inference import GenerationConfig, GenerationParams, generate_music

            lm_on = self.lm_ok and use_lm is not False
            kw: dict = dict(caption=caption, instrumental=instrumental, duration=float(duration),
                            inference_steps=INFERENCE_STEPS, seed=seed, thinking=lm_on,
                            lyrics=lyrics if (lyrics and not instrumental) else "[Instrumental]")
            if not lm_on:  # 言語モデルを完全に使わない (キャプション書き換えやメタ情報の推定もしない)
                kw.update(use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
            if bpm:
                kw["bpm"] = int(bpm)
            if keyscale:
                kw["keyscale"] = keyscale
            if vocal_language and not instrumental:
                kw["vocal_language"] = vocal_language
            if loop:
                kw.update(fade_in_duration=0.0, fade_out_duration=0.0)
            params = GenerationParams(**kw)
            config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[seed])

            tmp = OUTPUT_DIR / ".tmp" / uuid.uuid4().hex
            tmp.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                res = generate_music(self.dit, self.llm, params, config, save_dir=str(tmp))
                if not res.success or not res.audios:
                    raise RuntimeError(res.error or "生成に失敗しました")
                elapsed = round(time.time() - t0, 1)
                stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}"
                raw = OUTPUT_DIR / f"{stem}.wav"
                shutil.move(res.audios[0]["path"], raw)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                self.last_used = time.time()

        trim_info = trim_silence(raw) if (trim and not loop) else None
        item = dict(file=raw.name, path=str(raw), url=f"/outputs/{raw.name}", seed=seed, seconds=elapsed,
                    caption=caption, instrumental=instrumental, requested_duration=float(duration),
                    lm=LM_MODEL if lm_on else None, trim=trim_info,
                    time_costs=(res.extra_outputs or {}).get("time_costs"), footprint_gb=footprint_gb())
        if loop:
            looped = OUTPUT_DIR / f"{stem}-loop.wav"
            info = make_loop(raw, looped, crossfade=loop_crossfade)
            item.update(file=looped.name, path=str(looped), url=f"/outputs/{looped.name}",
                        raw_file=raw.name, loop=info)
        log.info("generated %s in %ss seed=%s %r", item["file"], elapsed, seed, caption[:60])
        return item

    def status(self) -> dict:
        return dict(
            loaded=self.loaded, loading=self.loading, error=self.load_error, lm_loaded=self.lm_ok,
            loaded_in_seconds=self.loaded_in, unloads=self.unloads,
            idle_seconds=round(time.time() - self.last_used, 1) if self.last_used else None,
            footprint_gb=footprint_gb(), settings=settings(),
        )


engine = MusicEngine()
