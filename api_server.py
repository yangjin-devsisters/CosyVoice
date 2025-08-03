# api_server.py
import sys
from fastapi import FastAPI, HTTPException
import torchaudio
import uvicorn
import os
import torch
from pydantic import BaseModel
from typing import Optional
import threading
import time
import logging
import signal

# --- 요청 데이터 구조 정의 ---
class TTSRequest(BaseModel):
    text: str
    output_filename: str
    absolute_path: Optional[str] = None
    prompt_speaker_path: Optional[str] = None
    instruction: Optional[str] = None

app = FastAPI()
server_instance = None 

# --- CosyVoice 모델 로드 ---
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

print("Loading CosyVoice2-0.5B model...")
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
print("Model loaded successfully.")

default_output_dir = "tts_outputs_fallback"
os.makedirs(default_output_dir, exist_ok=True)
default_prompt_path = './asset/zero_shot_prompt.wav'


@app.get("/status")
def get_status():
    return {"status": "running"}

@app.post("/shutdown")
async def shutdown_server():
    global server_instance
    if server_instance:
        server_instance.should_exit = True
        server_instance.force_exit = True
        await server_instance.shutdown()
    return {"message": "Server is shutting down."}

@app.post("/generate-tts")
async def generate_tts(request: TTSRequest):
    text = request.text
    output_filename = request.output_filename
    
    output_dir = request.absolute_path or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    app.logger.info("\n--- New TTS Request ---")
    app.logger.info(f"Text: {text}")
    app.logger.info(f"Filename: {output_filename}")
    app.logger.info(f"Instruction: {request.instruction}")
    
    prompt_text = ""
    
    if request.prompt_speaker_path and os.path.exists(request.prompt_speaker_path):
        app.logger.info(f"Using custom voice: {request.prompt_speaker_path}")
        prompt_speech_16k = load_wav(request.prompt_speaker_path, 16000)
    else:
        app.logger.info(f"Using default voice: {default_prompt_path}")
        prompt_speech_16k = load_wav(default_prompt_path, 16000)

    try:
        app.logger.info("Starting TTS inference...")
        
        if request.instruction:
            output_iterator = cosyvoice.inference_instruct2(request.text, request.instruction, prompt_speech_16k)
        else:
            output_iterator = cosyvoice.inference_zero_shot(request.text, prompt_text, prompt_speech_16k)
        
        saved = False
        for i, chunk in enumerate(output_iterator):
            save_path = os.path.join(output_dir, output_filename)
            app.logger.info(f"  - Saving chunk {i} to: {os.path.abspath(save_path)}")
            torchaudio.save(save_path, chunk['tts_speech'], cosyvoice.sample_rate)
            saved = True
        
        if not saved: app.logger.warning("No audio chunks were generated.")
        app.logger.info("Request processing finished.")
        return {"status": "success", "path": output_filename}

    except Exception as e:
        app.logger.error(f"[FATAL ERROR] An exception occurred: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

class Server(uvicorn.Server):
    def handle_exit(self, sig: int, frame) -> None:
        self.should_exit = True

def run_server():
    global server_instance
    config = uvicorn.Config(app, host="127.0.0.1", port=9881, log_level="info")
    server_instance = Server(config=config)
    
    app.logger = logging.getLogger("uvicorn.error")
    
    server_instance.run()

if __name__ == "__main__":
    run_server()