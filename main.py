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

app = FastAPI()


# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")
# Mount static files to serve index.html
# app.mount("/", StaticFiles(directory=".", html=True), name="static")



# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str
    history: list
    is_speech: bool = False

class ChatResponse(BaseModel):
    response: str
    audio: str = ""  # Default to empty string instead of None

# @app.get("/")
# async def root():
#     return renderHTML

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
        response = generate_response(validated_text, request.history)
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        # Initialize audio response
        audio_base64 = ""

        # Generate audio only for speech input
        if request.is_speech:
            try:
                audio_bytes = text_to_speech(response)
                if audio_bytes:
                    audio_base64 = base64.b64encode(audio_bytes).decode()
            except Exception as e:
                print(f"Audio generation error: {str(e)}")
                # Continue without audio if generation fails
                audio_base64 = ""

        return {"response": response, "audio": audio_base64}
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.options("/chat")
async def options_chat():
    return JSONResponse(content={}, status_code=200)