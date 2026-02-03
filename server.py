import os
import shutil
import tempfile
import torch
import soundfile as sf
import numpy as np
import uvicorn
import gc
import time
import asyncio
import yaml
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from qwen_tts import Qwen3TTSModel
from loguru import logger

# --- Configuration Loading ---
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# --- Logger Configuration ---
logger.add(
    config["logging"]["file"],
    level=config["logging"]["level"],
    rotation=config["logging"]["rotation"],
    retention=config["logging"]["retention"],
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(20)
logger.info(f"Using device: {DEVICE}")

# --- Model Configuration ---
MODELS = config["models"]
UNLOAD_TIMEOUT = config["server"]["unload_timeout"]

app = FastAPI(title="Qwen3-TTS Server")

# --- Global State for Memory Management ---
active_model = None
active_model_name = None
last_access_time = 0

def unload_model():
    """Unloads the currently active model and forces garbage collection."""
    global active_model, active_model_name
    if active_model is not None:
        logger.info(f"MEMORY MANAGER: Unloading model '{active_model_name}' to free RAM.")
        del active_model
        active_model = None
        active_model_name = None
        gc.collect()

def get_or_load_model(target_model_name: str):
    """
    Retrieves the requested model.
    1. If it's already loaded, updates timestamp and returns it.
    2. If a different model is loaded, unloads it first.
    3. Loads the requested model from disk.
    """
    global active_model, active_model_name, last_access_time
    
    last_access_time = time.time()

    if active_model_name == target_model_name and active_model is not None:
        return active_model

    if active_model is not None:
        logger.info(f"MEMORY MANAGER: Switching models. Unloading '{active_model_name}'...")
        unload_model()

    logger.info(f"MEMORY MANAGER: Loading '{target_model_name}' from disk...")
    try:
        model_id = MODELS[target_model_name]
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=DEVICE,
            torch_dtype=torch.float32
        )
            
        active_model = model
        active_model_name = target_model_name
        logger.info(f"MEMORY MANAGER: '{target_model_name}' loaded successfully.")
        return active_model
    except Exception as e:
        logger.error(f"CRITICAL ERROR loading model: {e}")
        raise e

async def inactivity_monitor():
    """Background task to unload models after inactivity."""
    global active_model, last_access_time
    logger.info("MEMORY MANAGER: Inactivity monitor started.")
    while True:
        await asyncio.sleep(10)
        if active_model is not None:
            elapsed = time.time() - last_access_time
            if elapsed > UNLOAD_TIMEOUT:
                logger.info(f"MEMORY MANAGER: Inactivity detected ({elapsed:.1f}s > {UNLOAD_TIMEOUT}s).")
                unload_model()

def process_audio_output(outputs):
    """
    Helper to extract audio and sample rate safely from model outputs.
    """
    audio_data, sample_rate = outputs["audio"], outputs["sample_rate"]

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().float().numpy()
    
    if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
        audio_data = audio_data.squeeze()
        
    return audio_data, int(sample_rate)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(inactivity_monitor())

@app.post("/voice-design")
async def voice_design(
    target_text: str = Form(...),
    language: str = Form(...),
    instruct: str = Form(...)
):
    model = get_or_load_model("1.7b-design")
    output_filename = f"output_design_{os.urandom(4).hex()}.wav"
    try:
        logger.info(f"Processing Voice Design request: '{instruct}'")
        outputs = model.generate_voice_design(
            text=target_text,
            language=language,
            instruct=instruct
        )
        audio_data, sample_rate = process_audio_output(outputs)
        sf.write(output_filename, audio_data, sample_rate)
        return FileResponse(output_filename, media_type="audio/wav", filename="design_output.wav")
    except Exception as e:
        logger.error(f"Error during voice design: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/custom-voice")
async def custom_voice(
    model_size: str = Form(...),
    language: str = Form(...),
    speaker: str = Form(...),
    instruct: str = Form(...),
    target_text: str = Form(...)
):
    model_name = f"{model_size}-custom"
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model size for custom voice.")
    
    model = get_or_load_model(model_name)
    output_filename = f"output_custom_{os.urandom(4).hex()}.wav"
    try:
        logger.info(f"Processing Custom Voice request for model: {model_name}")
        outputs = model.generate_custom_voice(
            text=target_text,
            language=language,
            speaker=speaker,
            instruct=instruct
        )
        audio_data, sample_rate = process_audio_output(outputs)
        sf.write(output_filename, audio_data, sample_rate)
        return FileResponse(output_filename, media_type="audio/wav", filename="custom_output.wav")
    except Exception as e:
        logger.error(f"Error during custom voice generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-clone")
async def voice_clone(
    model_size: str = Form(...),
    target_text: str = Form(...),
    language: str = Form(...),
    reference_text: str = Form(...),
    reference_audio: UploadFile = File(...)
):
    model_name = f"{model_size}-clone"
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model size for voice cloning.")
        
    model = get_or_load_model(model_name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
        shutil.copyfileobj(reference_audio.file, temp_ref)
        temp_ref_path = temp_ref.name

    output_filename = f"output_clone_{os.urandom(4).hex()}.wav"
    try:
        logger.info(f"Processing Voice Clone request for model: {model_name}")
        outputs = model.generate_voice_clone(
            text=target_text,
            language=language,
            ref_audio=temp_ref_path, 
            ref_text=reference_text
        )
        audio_data, sample_rate = process_audio_output(outputs)
        sf.write(output_filename, audio_data, sample_rate)
        return FileResponse(output_filename, media_type="audio/wav", filename="cloned_output.wav")
    except Exception as e:
        logger.error(f"Error during cloning: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config["server"]["host"],
        port=config["server"]["port"],
        timeout_keep_alive=config["server"]["request_timeout"]
    )
