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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from qwen_tts import Qwen3TTSModel

# --- Configuration ---
# Use CPU as requested. 
# Intel i9 with 20 cores: We set torch threads to match cores for faster inference.
DEVICE = "cpu"
torch.set_num_threads(20)

# Define Model Paths (HuggingFace IDs)
MODEL_DESIGN_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_CLONE_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
UNLOAD_TIMEOUT = 180  # Unload after 3 minutes of inactivity

app = FastAPI(title="Qwen3-TTS Server (LXC/CPU - Memory Efficient)")

# --- Global State for Memory Management ---
active_model = None
active_model_name = None  # "design" or "clone"
last_access_time = 0

def unload_model():
    """Unloads the currently active model and forces garbage collection."""
    global active_model, active_model_name
    if active_model is not None:
        print(f"MEMORY MANAGER: Unloading model '{active_model_name}' to free RAM.")
        del active_model
        active_model = None
        active_model_name = None
        # Force garbage collection to reclaim memory immediately
        gc.collect()

def get_or_load_model(target_model_name: str):
    """
    Retrieves the requested model.
    1. If it's already loaded, updates timestamp and returns it.
    2. If a different model is loaded, unloads it first (Mutual Exclusion).
    3. Loads the requested model from disk.
    """
    global active_model, active_model_name, last_access_time
    
    # Update access time
    last_access_time = time.time()

    # Case 1: The requested model is already in memory
    if active_model_name == target_model_name and active_model is not None:
        return active_model

    # Case 2: A different model is loaded. Unload it first to save RAM.
    if active_model is not None:
        print(f"MEMORY MANAGER: Switching models. Unloading '{active_model_name}'...")
        unload_model()

    # Case 3: Load the requested model
    print(f"MEMORY MANAGER: Loading '{target_model_name}' from disk...")
    try:
        if target_model_name == "design":
            model = Qwen3TTSModel.from_pretrained(
                MODEL_DESIGN_ID,
                device_map=DEVICE,
                torch_dtype=torch.float32
            )
        elif target_model_name == "clone":
            model = Qwen3TTSModel.from_pretrained(
                MODEL_CLONE_ID,
                device_map=DEVICE,
                torch_dtype=torch.float32
            )
        else:
            raise ValueError(f"Unknown model name: {target_model_name}")
            
        active_model = model
        active_model_name = target_model_name
        print(f"MEMORY MANAGER: '{target_model_name}' loaded successfully.")
        return active_model
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        raise e

async def inactivity_monitor():
    """Background task to unload models after inactivity."""
    global active_model, last_access_time
    print("MEMORY MANAGER: Inactivity monitor started.")
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        if active_model is not None:
            elapsed = time.time() - last_access_time
            if elapsed > UNLOAD_TIMEOUT:
                print(f"MEMORY MANAGER: Inactivity detected ({elapsed:.1f}s > {UNLOAD_TIMEOUT}s).")
                unload_model()

def process_audio_output(outputs):
    """
    Helper to extract audio and sample rate safely from model outputs.
    """
    audio_data = None
    sample_rate = None

    if isinstance(outputs, dict):
        audio_data = outputs.get("audio") or outputs.get("audio_data") or outputs.get("wav")
        sample_rate = outputs.get("sample_rate") or outputs.get("sampling_rate") or outputs.get("sr")
    elif isinstance(outputs, (list, tuple)):
        if len(outputs) >= 2:
            audio_data = outputs[0]
            sample_rate = outputs[1]
    
    if audio_data is None or sample_rate is None:
        raise ValueError(f"Could not parse model output. Type: {type(outputs)}, Content: {outputs}")

    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().float().numpy()
    elif isinstance(audio_data, list):
        # Explicitly convert list to numpy array to fix AttributeError: 'list' object has no attribute 'shape'
        audio_data = np.array(audio_data)
    
    if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
        audio_data = audio_data.squeeze()
        
    if hasattr(sample_rate, 'item'): 
        sample_rate = sample_rate.item()
    sample_rate = int(sample_rate)

    return audio_data, sample_rate

@app.on_event("startup")
async def startup_event():
    # Start the background monitor loop
    asyncio.create_task(inactivity_monitor())

@app.post("/voice-clone")
async def voice_clone(
    target_text: str = Form(...),
    language: str = Form(...),
    reference_text: str = Form(...),
    reference_audio: UploadFile = File(...)
):
    # Ensure model is loaded (this might take time if not already loaded)
    model = get_or_load_model("clone")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref:
        shutil.copyfileobj(reference_audio.file, temp_ref)
        temp_ref_path = temp_ref.name

    output_filename = f"output_clone_{os.urandom(4).hex()}.wav"

    try:
        print(f"Processing Voice Clone request for lang: {language}")
        
        outputs = model.generate_voice_clone(
            text=target_text,
            language=language,
            ref_audio=temp_ref_path, 
            ref_text=reference_text
        )
        
        audio_data, sample_rate = process_audio_output(outputs)
        
        print(f"Writing Audio Clone: Shape={audio_data.shape}, Rate={sample_rate}")
        sf.write(output_filename, audio_data, sample_rate)
        
        return FileResponse(output_filename, media_type="audio/wav", filename="cloned_output.wav")

    except Exception as e:
        print(f"Error during cloning: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)

@app.post("/voice-design")
async def voice_design(
    target_text: str = Form(...),
    language: str = Form(...),
    instruct: str = Form(...)
):
    # Ensure model is loaded
    model = get_or_load_model("design")

    output_filename = f"output_design_{os.urandom(4).hex()}.wav"

    try:
        print(f"Processing Voice Design request: '{instruct}'")
        
        outputs = model.generate_voice_design(
            text=target_text,
            language=language,
            instruct=instruct
        )
        
        audio_data, sample_rate = process_audio_output(outputs)
        
        print(f"Writing Audio Design: Shape={audio_data.shape}, Rate={sample_rate}")
        sf.write(output_filename, audio_data, sample_rate)
        
        return FileResponse(output_filename, media_type="audio/wav", filename="design_output.wav")

    except Exception as e:
        print(f"Error during voice design: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)