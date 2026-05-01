import os
import shutil
import hashlib
import tempfile
import torch
import soundfile as sf
import numpy as np
import uvicorn
import gc
import time
import asyncio
import yaml
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from qwen_tts import Qwen3TTSModel
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger.add(
    config["logging"]["file"],
    level=config["logging"]["level"],
    rotation=config["logging"]["rotation"],
    retention=config["logging"]["retention"],
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(20)
logger.info(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODELS = config["models"]
UNLOAD_TIMEOUT = config["server"]["unload_timeout"]

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

active_model      = None
active_model_name = None
last_access_time  = 0

# Tracks concurrent in-flight requests so the inactivity monitor never
# unloads the model while a generation is running.
# asyncio is single-threaded: plain int mutation is safe here because
# all mutations happen from the same event loop.
_requests_in_flight: int = 0

# Handle to the current background preload task (asyncio.Task), or None.
# Used by /status to report preload progress.
_preload_task_handle: Optional[asyncio.Task] = None

# ---------------------------------------------------------------------------
# Voice-clone prompt cache
# Key = MD5(reference_audio_bytes + reference_text)
# ---------------------------------------------------------------------------

_voice_prompt_cache: dict = {}

# ---------------------------------------------------------------------------
# Default voice references (persistent per model)
# Key = model_name (e.g., "1.7b-clone")
# ---------------------------------------------------------------------------

_default_voice_prompts: dict = {}  # Stores the actual voice_clone_prompt
_default_voice_metadata: dict = {}  # Stores {id, reference_text_preview}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(inactivity_monitor())
    yield


app = FastAPI(title="Qwen3-TTS Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Model memory management
# ---------------------------------------------------------------------------

def unload_model():
    """Unloads the active model and releases GPU memory."""
    global active_model, active_model_name
    if active_model is None:
        return

    logger.info(f"MEMORY MANAGER: Unloading model '{active_model_name}'.")

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024 ** 3
        mem_rsrv  = torch.cuda.memory_reserved()   / 1024 ** 3
        logger.info(
            f"MEMORY MANAGER: GPU before unload — "
            f"allocated {mem_alloc:.2f} GB, reserved {mem_rsrv:.2f} GB"
        )

    del active_model
    active_model      = None
    active_model_name = None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_alloc_after = torch.cuda.memory_allocated() / 1024 ** 3
        mem_rsrv_after  = torch.cuda.memory_reserved()   / 1024 ** 3
        logger.info(
            f"MEMORY MANAGER: GPU after unload — "
            f"allocated {mem_alloc_after:.2f} GB, reserved {mem_rsrv_after:.2f} GB"
        )

    logger.info("MEMORY MANAGER: Model unloaded.")


def get_or_load_model(target_model_name: str) -> Qwen3TTSModel:
    """
    Returns the requested model, loading (and optionally swapping) as needed.
    All callers must increment/decrement _requests_in_flight around this call.
    """
    global active_model, active_model_name, last_access_time

    last_access_time = time.time()

    if active_model_name == target_model_name and active_model is not None:
        return active_model

    if active_model is not None:
        logger.info(f"MEMORY MANAGER: Switching from '{active_model_name}' → '{target_model_name}'.")
        unload_model()

    logger.info(f"MEMORY MANAGER: Loading '{target_model_name}' from disk…")
    try:
        model_id = MODELS[target_model_name]
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=DEVICE,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        active_model      = model
        active_model_name = target_model_name
        logger.info(f"MEMORY MANAGER: '{target_model_name}' loaded.")
        return active_model
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model '{target_model_name}': {e}")
        raise


async def inactivity_monitor():
    """Unloads the model after a configurable idle period."""
    logger.info("MEMORY MANAGER: Inactivity monitor started.")
    while True:
        await asyncio.sleep(10)
        # Only unload when no request or preload is actively using the model.
        if active_model is not None and _requests_in_flight == 0:
            elapsed = time.time() - last_access_time
            if elapsed > UNLOAD_TIMEOUT:
                logger.info(
                    f"MEMORY MANAGER: Idle {elapsed:.1f}s "
                    f"(threshold {UNLOAD_TIMEOUT}s) — unloading."
                )
                unload_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_remove(path: str) -> None:
    """Best-effort file removal used as BackgroundTask after responses."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.warning(f"Could not remove temp file '{path}': {exc}")


def process_audio_output(outputs):
    """
    Extracts (audio_array, sample_rate) from model outputs.
    Handles dict, tuple, and list return formats for forward-compatibility.
    """
    if isinstance(outputs, dict):
        audio_data  = outputs["audio"]
        sample_rate = outputs["sample_rate"]
    elif isinstance(outputs, (tuple, list)):
        audio_data  = outputs[0]
        sample_rate = outputs[1] if len(outputs) > 1 else 12000
        if isinstance(audio_data, list) and audio_data:
            logger.debug(f"Audio data is a list ({len(audio_data)} elements) — extracting first.")
            audio_data = audio_data[0]
    else:
        raise TypeError(f"Unexpected model output type: {type(outputs)}")

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().float().numpy()

    if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
        audio_data = audio_data.squeeze()

    if isinstance(audio_data, np.ndarray):
        logger.debug(
            f"Audio — shape: {audio_data.shape}, "
            f"dtype: {audio_data.dtype}, sample_rate: {sample_rate}"
        )

    return audio_data, int(sample_rate)


def _gpu_warmup(model: Qwen3TTSModel, language: str = "en") -> None:
    """
    Runs a short silent/dummy generation to prime GPU CUDA kernels.

    The first inference call always triggers kernel compilation for new tensor
    shapes, which can add significant latency. Running a short dummy pass during
    preload absorbs this cost before the real request arrives.
    """
    logger.info("PRELOAD: Running GPU kernel warm-up pass…")
    t0 = time.perf_counter()
    try:
        _ = model.generate_voice_design(
            text="Warm up.",
            language=language,
            instruct="neutral voice",
        )
    except Exception as exc:
        # Warm-up is best-effort: if it fails, don't abort the preload.
        logger.warning(f"PRELOAD: Warm-up pass failed (non-fatal): {exc}")
    elapsed = time.perf_counter() - t0
    logger.info(f"PRELOAD: Warm-up completed in {elapsed:.2f}s — kernels are hot.")


# ---------------------------------------------------------------------------
# Preload background coroutine
# ---------------------------------------------------------------------------

async def _run_preload(
    model_name:     str,
    ref_bytes:      Optional[bytes],
    reference_text: Optional[str],
    run_warmup:     bool,
    warmup_language: str,
):
    """
    Coroutine scheduled by /preload.

    Execution sequence:
      1. Load (or confirm already loaded) the requested model.
      2. If reference audio + text supplied → pre-encode the voice-clone prompt
         and store it in _voice_prompt_cache (keyed by MD5 of bytes + text).
      3. If run_warmup=True → run a short dummy inference to prime GPU kernels.

    Because this is an asyncio coroutine on a single-threaded event loop,
    blocking calls here will block the loop temporarily — but the 202 response
    has already been sent to the client, who is free to do other work (e.g.,
    LLM text generation) in parallel.
    """
    global _requests_in_flight
    _requests_in_flight += 1
    try:
        # ---- 1. Model load ------------------------------------------------
        t_load = time.perf_counter()
        model = get_or_load_model(model_name)
        logger.info(
            f"PRELOAD: '{model_name}' ready "
            f"({time.perf_counter() - t_load:.2f}s)."
        )

        # ---- 2. Voice-clone prompt pre-encoding ---------------------------
        if ref_bytes and reference_text and model_name.endswith("-clone"):
            cache_key = hashlib.md5(ref_bytes + reference_text.encode()).hexdigest()
            if cache_key not in _voice_prompt_cache:
                logger.info(f"PRELOAD: Encoding voice prompt [{cache_key[:8]}…]")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(ref_bytes)
                    tmp_path = tmp.name
                try:
                    t_enc = time.perf_counter()
                    _voice_prompt_cache[cache_key] = model.create_voice_clone_prompt(
                        ref_audio=tmp_path,
                        ref_text=reference_text,
                    )
                    logger.info(
                        f"PRELOAD: Voice prompt cached [{cache_key[:8]}…] "
                        f"({time.perf_counter() - t_enc:.2f}s)."
                    )
                finally:
                    _safe_remove(tmp_path)
            else:
                logger.info(f"PRELOAD: Voice prompt already cached [{cache_key[:8]}…].")

        # ---- 3. GPU kernel warm-up ----------------------------------------
        if run_warmup:
            _gpu_warmup(model, language=warmup_language)

        logger.info(f"PRELOAD: All steps complete for '{model_name}'. Server is ready.")

    except Exception as exc:
        logger.error(f"PRELOAD: Failed for '{model_name}': {exc}")
    finally:
        _requests_in_flight -= 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/preload", status_code=202)
async def preload(
    model_type:      str           = Form(...),
    model_size:      str           = Form(default="1.7b"),
    reference_text:  Optional[str] = Form(default=None),
    reference_audio: Optional[UploadFile] = File(default=None),
    warmup:          bool          = Form(default=True),
    warmup_language: str           = Form(default="en"),
):
    """
    Pre-loads the model (and optionally pre-encodes a voice-clone prompt) so
    that the actual TTS call incurs zero model-load latency.

    Intended usage (client side):
      1. Send POST /preload as soon as you know you'll need TTS.
      2. Do your LLM text generation in parallel.
      3. Send the actual /voice-clone (or /custom-voice) request — the model
         is already loaded and warm.

    Parameters
    ----------
    model_type : "clone" | "custom" | "design"
    model_size : "0.6b" | "1.7b"  (ignored for model_type=design)
    reference_text  : transcript of reference audio (voice-clone only)
    reference_audio : reference WAV file (voice-clone only)
        If both are supplied, the voice-clone prompt is pre-encoded and cached.
        The subsequent /voice-clone call MUST use the same reference audio bytes
        and reference_text for the cache to hit.
    warmup : run a short dummy inference to prime GPU kernels (default: true)
    warmup_language : language code for the warm-up pass (default: "en")
    """
    global _preload_task_handle

    # Build the model name, mirroring the pattern used by generation endpoints.
    if model_type == "design":
        model_name = "1.7b-design"
    else:
        model_name = f"{model_size}-{model_type}"

    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown combination: model_type='{model_type}', model_size='{model_size}'.",
        )

    # Read reference audio bytes upfront (UploadFile is stream-backed).
    ref_bytes = await reference_audio.read() if reference_audio else None

    # If the same model is already loaded and no voice prompt is pending, skip.
    prompt_needed = (
        ref_bytes
        and reference_text
        and model_name.endswith("-clone")
        and hashlib.md5(ref_bytes + reference_text.encode()).hexdigest()
        not in _voice_prompt_cache
    )
    if active_model_name == model_name and active_model is not None and not prompt_needed:
        logger.info(f"PRELOAD: '{model_name}' already loaded — nothing to do.")
        return {"status": "already_ready", "model": model_name}

    # Cancel any existing preload task that is no longer relevant.
    if _preload_task_handle and not _preload_task_handle.done():
        logger.info("PRELOAD: Cancelling previous preload task.")
        _preload_task_handle.cancel()

    # Schedule the preload coroutine. The 202 response is sent to the client
    # before this task begins executing, so the client is free to proceed.
    _preload_task_handle = asyncio.create_task(
        _run_preload(model_name, ref_bytes, reference_text, warmup, warmup_language)
    )
    _preload_task_handle.add_done_callback(
        lambda t: logger.error(f"PRELOAD task raised: {t.exception()}")
        if not t.cancelled() and t.exception()
        else None
    )

    logger.info(f"PRELOAD: Task scheduled for '{model_name}'.")
    return {
        "status":      "loading",
        "model":       model_name,
        "warmup":      warmup,
        "voice_prompt_preload": bool(ref_bytes and reference_text),
    }


@app.get("/status")
async def status():
    """
    Returns the current server state: which model is loaded, whether a preload
    is in progress, how many requests are in flight, cache statistics, and
    default reference information.
    """
    preload_state = "idle"
    if _preload_task_handle is not None:
        if not _preload_task_handle.done():
            preload_state = "loading"
        elif _preload_task_handle.cancelled():
            preload_state = "cancelled"
        elif _preload_task_handle.exception():
            preload_state = "failed"
        else:
            preload_state = "ready"

    # Build default references info
    default_refs = {}
    for model_name, metadata in _default_voice_metadata.items():
        default_refs[model_name] = {
            "id": metadata["id"],
            "reference_text_preview": metadata["reference_text_preview"]
        }

    return {
        "device":               DEVICE,
        "model_loaded":         active_model is not None,
        "model_name":           active_model_name,
        "preload_state":        preload_state,
        "requests_in_flight":   _requests_in_flight,
        "voice_prompts_cached": len(_voice_prompt_cache),
        "default_references":   default_refs,
    }


@app.post("/voice-design")
async def voice_design(
    background_tasks: BackgroundTasks,
    target_text: str = Form(...),
    language:    str = Form(...),
    instruct:    str = Form(...),
):
    """
    Generates a voice from a text description.
    NOTE: The 1.7B model is used unconditionally — Qwen3-TTS does not
    publish a 0.6B voice-design variant.
    """
    global _requests_in_flight
    _requests_in_flight += 1
    output_filename = f"output_design_{os.urandom(4).hex()}.wav"
    background_tasks.add_task(_safe_remove, output_filename)
    try:
        model = get_or_load_model("1.7b-design")
        logger.info(f"Voice Design — instruct: '{instruct}'")
        outputs = model.generate_voice_design(
            text=target_text,
            language=language,
            instruct=instruct,
        )
        audio_data, sample_rate = process_audio_output(outputs)
        sf.write(output_filename, audio_data, sample_rate)
        return FileResponse(output_filename, media_type="audio/wav", filename="design_output.wav")
    except Exception as exc:
        logger.error(f"Voice Design error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _requests_in_flight -= 1


@app.post("/custom-voice")
async def custom_voice(
    background_tasks: BackgroundTasks,
    model_size:  str = Form(...),
    language:    str = Form(...),
    speaker:     str = Form(...),
    instruct:    str = Form(...),
    target_text: str = Form(...),
):
    model_name = f"{model_size}-custom"
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model size '{model_size}' for custom voice.",
        )

    global _requests_in_flight
    _requests_in_flight += 1
    output_filename = f"output_custom_{os.urandom(4).hex()}.wav"
    background_tasks.add_task(_safe_remove, output_filename)
    try:
        model = get_or_load_model(model_name)
        logger.info(f"Custom Voice — model: {model_name}, speaker: '{speaker}'")
        outputs = model.generate_custom_voice(
            text=target_text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
        audio_data, sample_rate = process_audio_output(outputs)
        sf.write(output_filename, audio_data, sample_rate)
        return FileResponse(output_filename, media_type="audio/wav", filename="custom_output.wav")
    except Exception as exc:
        logger.error(f"Custom Voice error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _requests_in_flight -= 1


@app.post("/voice-clone/set-reference")
async def set_default_reference(
    model_size:      str        = Form(...),
    reference_text:  str        = Form(...),
    reference_audio: UploadFile = File(...),
):
    """
    Set a default reference voice for the specified model that will be used
    for all subsequent /voice-clone calls that don't provide their own reference.
    
    This eliminates the need to upload reference audio on every voice-clone request.
    
    Parameters
    ----------
    model_size : "0.6b" | "1.7b"
    reference_text : transcript of the reference audio
    reference_audio : reference WAV file (3-10 seconds recommended)
    
    Returns
    -------
    JSON with status, reference_id, and confirmation message
    """
    model_name = f"{model_size}-clone"
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model size '{model_size}' for voice cloning.",
        )
    
    global _requests_in_flight
    _requests_in_flight += 1
    
    try:
        # Load the model
        model = get_or_load_model(model_name)
        
        # Read reference audio bytes
        ref_bytes = await reference_audio.read()
        
        # Generate a short ID for this reference
        ref_id = hashlib.md5(ref_bytes + reference_text.encode()).hexdigest()[:8]
        
        # Create voice clone prompt
        logger.info(f"Setting default reference for '{model_name}' [ID: {ref_id}]")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(ref_bytes)
            tmp_path = tmp.name
        
        try:
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=tmp_path,
                ref_text=reference_text,
            )
            
            # Store the default voice prompt and metadata
            _default_voice_prompts[model_name] = voice_prompt
            _default_voice_metadata[model_name] = {
                "id": ref_id,
                "reference_text_preview": reference_text[:50] + ("..." if len(reference_text) > 50 else "")
            }
            
            logger.info(f"Default reference set for '{model_name}' [ID: {ref_id}]")
            
            return {
                "status": "success",
                "model": model_name,
                "reference_id": ref_id,
                "message": f"Default reference set for {model_name} model"
            }
        finally:
            _safe_remove(tmp_path)
            
    except Exception as exc:
        logger.error(f"Set default reference error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _requests_in_flight -= 1


@app.post("/voice-clone")
async def voice_clone(
    background_tasks: BackgroundTasks,
    model_size:      str                = Form(...),
    target_text:     str                = Form(...),
    language:        str                = Form(...),
    reference_text:  Optional[str]      = Form(default=None),
    reference_audio: Optional[UploadFile] = File(default=None),
):
    """
    Clone a voice and generate speech. Reference audio/text can be provided inline
    or omitted to use the default reference set via /voice-clone/set-reference.
    
    Parameters
    ----------
    model_size : "0.6b" | "1.7b"
    target_text : text to synthesize in the cloned voice
    language : language code (e.g., "en", "zh")
    reference_text : (optional) transcript of reference audio. If omitted, uses default.
    reference_audio : (optional) reference audio file. If omitted, uses default.
    
    Behavior
    --------
    - Both reference_audio and reference_text provided → use them (override default)
    - Neither provided → use default reference (must be set first)
    - Only one provided → error (must provide both or neither)
    """
    model_name = f"{model_size}-clone"
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model size '{model_size}' for voice cloning.",
        )

    # Determine if using provided reference or default
    has_ref_audio = reference_audio is not None
    has_ref_text = reference_text is not None
    
    # Validate: must provide both or neither
    if has_ref_audio != has_ref_text:
        raise HTTPException(
            status_code=400,
            detail="Must provide both reference_audio and reference_text, or neither (to use default)."
        )
    
    use_default = not has_ref_audio and not has_ref_text
    
    global _requests_in_flight
    _requests_in_flight += 1
    output_filename = f"output_clone_{os.urandom(4).hex()}.wav"
    background_tasks.add_task(_safe_remove, output_filename)
    
    try:
        model = get_or_load_model(model_name)

        # ------------------------------------------------------------------
        # Determine voice prompt source
        # ------------------------------------------------------------------
        if use_default:
            # Use default reference
            if model_name not in _default_voice_prompts:
                raise HTTPException(
                    status_code=400,
                    detail=f"No default reference set for {model_name}. Use /voice-clone/set-reference first."
                )
            
            voice_prompt = _default_voice_prompts[model_name]
            ref_id = _default_voice_metadata[model_name]["id"]
            logger.info(f"Voice Clone — using default reference [ID: {ref_id}] for {model_name}")
        
        else:
            # Use provided reference (inline)
            ref_bytes = await reference_audio.read()
            cache_key = hashlib.md5(ref_bytes + reference_text.encode()).hexdigest()
            
            if cache_key not in _voice_prompt_cache:
                logger.info(f"Voice Clone — cache miss [{cache_key[:8]}…], encoding reference.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(ref_bytes)
                    tmp_path = tmp.name
                try:
                    _voice_prompt_cache[cache_key] = model.create_voice_clone_prompt(
                        ref_audio=tmp_path,
                        ref_text=reference_text,
                    )
                finally:
                    _safe_remove(tmp_path)
            else:
                logger.info(f"Voice Clone — cache hit [{cache_key[:8]}…].")
            
            voice_prompt = _voice_prompt_cache[cache_key]

        # ------------------------------------------------------------------
        # Generate speech
        # ------------------------------------------------------------------
        logger.info(f"Voice Clone — model: {model_name}, generating…")
        outputs = model.generate_voice_clone(
            text=target_text,
            language=language,
            voice_clone_prompt=voice_prompt,
        )
        audio_data, sample_rate = process_audio_output(outputs)
        sf.write(output_filename, audio_data, sample_rate)
        return FileResponse(output_filename, media_type="audio/wav", filename="cloned_output.wav")
        
    except Exception as exc:
        logger.error(f"Voice Clone error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _requests_in_flight -= 1


@app.delete("/voice-clone/cache")
async def clear_voice_clone_cache(clear_defaults: bool = False):
    """
    Clears the in-memory voice-clone prompt cache. Optionally clears default
    references as well.
    
    Parameters
    ----------
    clear_defaults : bool (query parameter)
        If True, also clears all default references set via /voice-clone/set-reference
        Default: False
    
    Example
    -------
    DELETE /voice-clone/cache              # Clear only cache
    DELETE /voice-clone/cache?clear_defaults=true  # Clear cache and defaults
    """
    cache_count = len(_voice_prompt_cache)
    _voice_prompt_cache.clear()
    
    defaults_count = 0
    if clear_defaults:
        defaults_count = len(_default_voice_prompts)
        _default_voice_prompts.clear()
        _default_voice_metadata.clear()
        logger.info(f"Voice Clone cache and defaults cleared ({cache_count} cached, {defaults_count} defaults removed).")
        return {
            "cache_cleared": cache_count,
            "defaults_cleared": defaults_count
        }
    else:
        logger.info(f"Voice Clone cache cleared ({cache_count} entry/entries removed).")
        return {
            "cache_cleared": cache_count,
            "defaults_cleared": 0
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config["server"]["host"],
        port=config["server"]["port"],
        timeout_keep_alive=config["server"]["request_timeout"],
    )