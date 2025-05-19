from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from llm import generate_response
from tts import text_to_speech
from stt import validate_speech_input
import base64
from fastapi.templating import Jinja2Templates
import os
from typing import List, Optional

app = FastAPI()

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str
    history: list
    is_speech: bool = False
    dual_response: bool = False

class ChatResponse(BaseModel):
    responses: List[str]
    audio: Optional[str] = None

class AudioRequest(BaseModel):
    text: str

class AudioResponse(BaseModel):
    audio: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Validate speech input
        validated_text = validate_speech_input(request.text)
        if not validated_text:
            raise HTTPException(status_code=400, detail="Invalid speech input")

        # Generate response
        response_data = generate_response(
            validated_text, 
            request.history, 
            dual_response=request.dual_response
        )
        
        if not response_data or not response_data.get("responses"):
            raise HTTPException(status_code=500, detail="Failed to generate response")

        # Initialize audio response
        audio_base64 = None

        # Generate audio only for speech input and if NOT in dual response mode
        if request.is_speech and not request.dual_response and response_data["responses"]:
            try:
                # Use the first response for audio
                audio_bytes = text_to_speech(response_data["responses"][0])
                if audio_bytes:
                    audio_base64 = base64.b64encode(audio_bytes).decode()
            except Exception as e:
                print(f"Audio generation error: {str(e)}")
                # Continue without audio if generation fails

        return {
            "responses": response_data["responses"],
            "audio": audio_base64
        }
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_audio", response_model=AudioResponse)
async def generate_audio(request: AudioRequest):
    try:
        # Generate audio for the selected response
        audio_bytes = text_to_speech(request.text)
        if audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode()
            return {"audio": audio_base64}
        else:
            return {"audio": None}
    except Exception as e:
        print(f"Audio generation error: {str(e)}")
        return {"audio": None}

@app.options("/chat")
async def options_chat():
    return JSONResponse(content={}, status_code=200)

@app.options("/generate_audio")
async def options_generate_audio():
    return JSONResponse(content={}, status_code=200)