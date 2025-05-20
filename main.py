from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from llm import LLMHandler
from pdf_processor import PDFProcessor
from rag import RAGSystem
from fastapi.templating import Jinja2Templates
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import uvicorn
import json
import base64
from tts import text_to_speech  # Import the TTS function
from elevenlabs import ElevenLabs, VoiceSettings
from io import BytesIO
import asyncio

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", description="RAG System API for PDF Processing and Querying")

# Load API keys from environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY not set in environment variables")
    raise RuntimeError("ELEVENLABS_API_KEY not set")

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Create necessary directories
DATA_DIR = Path("DATA")
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Create directories if they don't exist
RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Verify embeddings, exclude .gitkeep
embedding_dirs = [d for d in EMBEDDINGS_DIR.glob("*") if d.is_dir() and d.name != ".gitkeep"]
if embedding_dirs:
    logger.info(f"Found {len(embedding_dirs)} existing embedding directories:")
    for dir in embedding_dirs:
        logger.info(f"- {dir.name}")
else:
    logger.warning("No existing embeddings found!")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system components
try:
    logger.info("Initializing RAG system components...")
    pdf_processor = PDFProcessor(data_dir="DATA")
    llm_handler = LLMHandler(model_name="gpt-3.5-turbo")
    rag_system = RAGSystem(pdf_processor, llm_handler)
    
    # Load existing embeddings
    if embedding_dirs:
        logger.info("Loading existing embeddings...")
        if rag_system.process_documents():
            logger.info("Successfully loaded existing embeddings")
        else:
            logger.warning("Failed to load existing embeddings")
    else:
        logger.warning("No embeddings found to load")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    raise

class Message(BaseModel):
    role: str
    text: str
    timestamp: Optional[str] = None
    isSpeech: Optional[bool] = False

class ChatRequest(BaseModel):
    text: str
    history: Optional[List[Message]] = []
    is_speech: Optional[bool] = False
    dual_response: Optional[bool] = False

class ChatResponse(BaseModel):
    responses: List[str]
    audio: Optional[str] = None

# Streaming audio generator
async def stream_audio(text: str):
    try:
        # Use ElevenLabs streaming API
        audio_stream = await elevenlabs_client.text_to_speech_stream(
            text=text,
            voice="Rachel",  # Replace with your preferred voice ID
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )

        # Yield audio chunks as they are received
        async for chunk in audio_stream:
            yield chunk
    except Exception as e:
        logger.error(f"Error streaming audio from ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error streaming audio: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    logger.info("Serving index page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.text}, dual_response: {request.dual_response}")
        
        # Check if we have any processed documents
        if not rag_system.documents_processed:
            return {
                "responses": ["No documents have been processed. Please upload a PDF first."],
                "audio": None
            }

        # Query the RAG system
        context = rag_system.query(request.text)
        if not context:
            logger.warning("No response generated for query")
            return {
                "responses": ["No relevant information found in the documents. Please try a different query."],
                "audio": None
            }
        
        # Generate response using RAG system
        response = rag_system.generate_response(
            request.text, 
            context, 
            dual_response=request.dual_response
        )
        
        # Generate audio if the request is from speech input
        audio_data = None
        if request.is_speech and not request.dual_response:
            # For single response, we'll stream the audio separately
            pass  # Audio will be streamed via /stream_audio endpoint
        
        logger.info("Successfully generated response")
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return {
            "responses": ["I apologize, but I encountered an error. Please try again."],
            "audio": None
        }

@app.post("/stream_audio")
async def stream_audio_endpoint(request: ChatRequest):
    try:
        # Use the first response for audio (since dual_response should be false here)
        text = request.text if request.dual_response else request.history[-1].text
        return StreamingResponse(
            stream_audio(text),
            media_type="audio/mpeg"
        )
    except Exception as e:
        logger.error(f"Error in stream_audio endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save the uploaded file
        file_path = RAW_PDFS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        if rag_system.process_document(str(file_path)):
            return {"message": f"Successfully processed {file.filename}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
            
    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Start the FastAPI server with uvicorn"""
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start()