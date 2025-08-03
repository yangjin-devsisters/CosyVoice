# ... (상단 import 구문들은 이전과 동일) ...
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
import logging # 로깅 라이브러리 추가

# --- 요청 데이터 구조 정의 ---
class TTSRequest(BaseModel):
    text: str
    output_filename: str
    absolute_path: Optional[str] = None

shutdown_event = threading.Event()
app = FastAPI()

# --- CosyVoice 모델 로드 ---
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

print("Loading CosyVoice2-0.5B model from local path...")
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
print("Model loaded successfully.")

default_output_dir = "tts_outputs_fallback"
os.makedirs(default_output_dir, exist_ok=True)


# --- API 엔드포인트들 (이전과 동일) ---
@app.get("/status")
def get_status():
    return {"status": "running"}

@app.post("/shutdown")
def shutdown_server():
    shutdown_event.set()
    return {"message": "Server is shutting down."}

@app.post("/generate-tts")
async def generate_tts(request: TTSRequest):
    # ... (generate-tts 함수 내용은 이전과 동일) ...
    text = request.text
    output_filename = request.output_filename
    
    if request.absolute_path:
        output_dir = request.absolute_path
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = default_output_dir
    
    app.logger.info("\n--- New TTS Request ---")
    app.logger.info(f"[1/5] Received Text: {text}")
    app.logger.info(f"[2/5] Received Filename: {output_filename}")
    app.logger.info(f"[INFO] Target directory: {output_dir}")
    
    prompt_text = "안녕하세요, 제 목소리 어때요?"
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

    try:
        app.logger.info("[3/5] Starting TTS inference...")
        output_iterator = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)
        
        app.logger.info("[4/5] Iterating through inference output...")
        saved = False
        for i, chunk in enumerate(output_iterator):
            app.logger.info(f"  - Found chunk {i}. Shape: {chunk['tts_speech'].shape}")
            save_path = os.path.join(output_dir, output_filename)
            app.logger.info(f"  - Attempting to save to: {os.path.abspath(save_path)}")
            torchaudio.save(save_path, chunk['tts_speech'], cosyvoice.sample_rate)
            app.logger.info(f"  - File saved successfully.")
            saved = True
        
        if not saved: app.logger.warning("[WARNING] Inference completed, but no audio chunks were generated or saved.")
        app.logger.info("[5/5] Request processing finished.")
        return {"status": "success", "path": output_filename}

    except Exception as e:
        app.logger.error(f"[FATAL ERROR] An exception occurred: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# --- 서버 실행 및 종료 로직 수정 ---
def run_server():
    # Uvicorn의 로그 설정을 변경하여 INFO 레벨은 stdout으로, ERROR는 stderr로 보내도록 설정
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host="127.0.0.1", port=9881, log_config=log_config)

if __name__ == "__main__":
    # FastAPI 앱에 로거 추가
    app.logger = logging.getLogger("uvicorn.error")
    
    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    print("Server started in a background thread.")
    
    shutdown_event.wait()
    print("Shutting down server.")