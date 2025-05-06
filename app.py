import os
import io
import torch
import numpy as np
import uvicorn
import asyncio
import soundfile as sf
import librosa
import time
import threading
import queue
import json
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request, Form, Query, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kasayie-asr")

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="KasaYie ASR API", 
    description="Real-time ASR API for Akan speech with support for speech impairments",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates for HTML pages
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root route
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Demo page route
@app.get("/demo", response_class=HTMLResponse)
async def demo():
    return FileResponse("static/demo.html")

# Create index.html in templates directory
with open("templates/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>KasaYie ASR API</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px;
                line-height: 1.6;
                color: #333;
            }
            h1 { color: #4f46e5; margin-bottom: 0.5em; }
            h2 { color: #4338ca; margin-top: 1.5em; }
            h3 { color: #3730a3; }
            code { 
                background-color: #f0f0f0; 
                padding: 2px 4px; 
                border-radius: 3px;
                font-family: 'Courier New', Courier, monospace;
            }
            pre { 
                background-color: #f0f0f0; 
                padding: 15px; 
                border-radius: 5px; 
                overflow-x: auto;
                border-left: 4px solid #4f46e5;
            }
            .endpoint { 
                margin-bottom: 30px;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #e5e7eb;
                background-color: #f9fafb;
            }
            .method {
                display: inline-block;
                padding: 3px 6px;
                border-radius: 4px;
                margin-right: 8px;
                font-weight: bold;
                color: white;
            }
            .post { background-color: #10b981; }
            .get { background-color: #3b82f6; }
            .ws { background-color: #8b5cf6; }
            .param {
                margin-bottom: 10px;
                padding-left: 10px;
                border-left: 3px solid #d1d5db;
            }
            .param-name {
                font-weight: bold;
            }
            .demo-link {
                display: inline-block;
                margin-top: 20px;
                padding: 10px 15px;
                background-color: #4f46e5;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
            }
            .demo-link:hover {
                background-color: #4338ca;
            }
            footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e5e7eb;
                text-align: center;
                font-size: 0.9em;
                color: #6b7280;
            }
        </style>
    </head>
    <body>
        <h1>KasaYie ASR API</h1>
        <p>Welcome to the KasaYie ASR API for Akan speech with support for speech impairments.</p>
        
        <a href="/demo" class="demo-link">Try the Demo</a>
        
        <h2>API Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method post">POST</span> /api/transcribe</h3>
            <p>Upload an audio file for transcription.</p>
            
            <h4>Parameters:</h4>
            <div class="param">
                <div class="param-name">file</div>
                <div>Audio file (WAV, MP3, etc.)</div>
            </div>
            <div class="param">
                <div class="param-name">language</div>
                <div>Language code: 'ak' for Akan/Twi (default), 'en' for English</div>
            </div>
            <div class="param">
                <div class="param-name">task</div>
                <div>'transcribe' (default) to output in original language, 'translate' to translate to English</div>
            </div>
            
            <h4>Example:</h4>
            <pre>curl -X POST -F "file=@audio.wav" -F "language=ak" -F "task=transcribe" http://localhost:8000/api/transcribe</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method ws">WebSocket</span> /api/stream</h3>
            <p>Stream audio in real-time for continuous transcription.</p>
            
            <h4>Parameters:</h4>
            <div class="param">
                <div class="param-name">language</div>
                <div>Language code: 'ak' for Akan/Twi (default), 'en' for English</div>
            </div>
            <div class="param">
                <div class="param-name">task</div>
                <div>'transcribe' (default) to output in original language, 'translate' to translate to English</div>
            </div>
            
            <h4>Example:</h4>
            <pre>
// JavaScript example
const socket = new WebSocket('ws://localhost:8000/api/stream?language=ak&task=transcribe');
socket.onopen = () => {
  // Start sending audio data
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = e => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(e.data);
        }
      };
      recorder.start(250); // Send data every 250ms
    });
};
socket.onmessage = event => {
  const response = JSON.parse(event.data);
  if (response.type === 'transcription') {
    console.log('Transcription:', response.text);
  }
};</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method get">GET</span> /health</h3>
            <p>Check the health status of the API.</p>
            <h4>Example:</h4>
            <pre>curl http://localhost:8000/health</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method get">GET</span> /stats</h3>
            <p>Get statistics about API usage.</p>
            <h4>Example:</h4>
            <pre>curl http://localhost:8000/stats</pre>
        </div>
        
        <footer>
            KasaYie ASR API - Powered by Whisper for Akan/Twi Speech Recognition
        </footer>
    </body>
    </html>
    """)

# Create demo.html in static directory
with open("static/demo.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>KasaYie ASR Demo</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
                color: #333;
            }
            h1 { color: #4f46e5; }
            button {
                padding: 10px 15px;
                background-color: #4f46e5;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
            }
            button:hover { background-color: #4338ca; }
            button:disabled {
                background-color: #9ca3af;
                cursor: not-allowed;
            }
            .controls {
                display: flex;
                gap: 10px;
                align-items: center;
                margin-bottom: 20px;
            }
            .status {
                margin-left: 15px;
                font-size: 14px;
                color: #4b5563;
            }
            .transcript-container {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 5px;
                padding: 15px;
                min-height: 200px;
                margin-bottom: 20px;
            }
            .transcript {
                white-space: pre-wrap;
            }
            select {
                padding: 8px;
                border-radius: 5px;
                border: 1px solid #d1d5db;
                margin-right: 10px;
            }
            .upload-container {
                margin-top: 30px;
                padding: 20px;
                border: 2px dashed #d1d5db;
                border-radius: 5px;
                text-align: center;
            }
            .upload-container input {
                display: none;
            }
            .upload-label {
                display: inline-block;
                padding: 10px 15px;
                background-color: #4f46e5;
                color: white;
                border-radius: 5px;
                cursor: pointer;
            }
            .upload-label:hover {
                background-color: #4338ca;
            }
            .settings {
                margin-top: 20px;
                padding: 15px;
                background-color: #f3f4f6;
                border-radius: 5px;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-connected { background-color: #10b981; }
            .status-disconnected { background-color: #ef4444; }
            .status-connecting { background-color: #f59e0b; }
            .visualizer {
                width: 100%;
                height: 100px;
                background-color: #f3f4f6;
                margin-bottom: 20px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>KasaYie ASR Demo</h1>
        <p>Test the Akan/Twi speech recognition with support for speech impairments</p>
        
        <div class="settings">
            <label for="language">Language:</label>
            <select id="language">
                <option value="ak" selected>Akan/Twi</option>
                <option value="en">English</option>
            </select>
            
            <label for="task">Task:</label>
            <select id="task">
                <option value="transcribe" selected>Transcribe</option>
                <option value="translate">Translate to English</option>
            </select>
        </div>
        
        <h2>Real-time Transcription</h2>
        <canvas id="visualizer" class="visualizer"></canvas>
        
        <div class="controls">
            <button id="startBtn">Start Recording</button>
            <button id="stopBtn" disabled>Stop Recording</button>
            <div class="status">
                <span class="status-indicator status-disconnected" id="statusIndicator"></span>
                <span id="statusText">Disconnected</span>
            </div>
        </div>
        
        <div class="transcript-container">
            <div class="transcript" id="transcript"></div>
        </div>
        
        <h2>Upload Audio File</h2>
        <div class="upload-container">
            <input type="file" id="audioFileInput" accept="audio/*" />
            <label for="audioFileInput" class="upload-label">Choose Audio File</label>
            <p id="fileName">No file chosen</p>
            <button id="uploadBtn" disabled>Transcribe File</button>
        </div>
        
        <script>
            // DOM Elements
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const transcript = document.getElementById('transcript');
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            const languageSelect = document.getElementById('language');
            const taskSelect = document.getElementById('task');
            const audioFileInput = document.getElementById('audioFileInput');
            const fileName = document.getElementById('fileName');
            const uploadBtn = document.getElementById('uploadBtn');
            const visualizer = document.getElementById('visualizer');
            const visualizerCtx = visualizer.getContext('2d');
            
            // Variables
            let socket;
            let mediaRecorder;
            let audioStream;
            let isRecording = false;
            let audioContext;
            let analyzer;
            let dataArray;
            let drawVisual;
            
            // Set up visualizer
            function setupVisualizer(stream) {
                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                
                const source = audioContext.createMediaStreamSource(stream);
                analyzer = audioContext.createAnalyser();
                analyzer.fftSize = 256;
                source.connect(analyzer);
                
                const bufferLength = analyzer.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);
                
                visualizer.width = visualizer.offsetWidth;
                visualizer.height = visualizer.offsetHeight;
                
                function draw() {
                    drawVisual = requestAnimationFrame(draw);
                    
                    analyzer.getByteFrequencyData(dataArray);
                    
                    visualizerCtx.fillStyle = '#f3f4f6';
                    visualizerCtx.fillRect(0, 0, visualizer.width, visualizer.height);
                    
                    const barWidth = (visualizer.width / bufferLength) * 2.5;
                    let x = 0;
                    
                    for (let i = 0; i < bufferLength; i++) {
                        const barHeight = dataArray[i] / 2;
                        
                        visualizerCtx.fillStyle = `rgb(${barHeight + 100}, 79, 229)`;
                        visualizerCtx.fillRect(x, visualizer.height - barHeight, barWidth, barHeight);
                        
                        x += barWidth + 1;
                    }
                }
                
                draw();
            }
            
            // WebSocket Connection
            function connectWebSocket() {
                const language = languageSelect.value;
                const task = taskSelect.value;
                
                // Update status
                statusIndicator.className = 'status-indicator status-connecting';
                statusText.textContent = 'Connecting...';
                
                // Create WebSocket connection
                socket = new WebSocket(`ws://${window.location.host}/api/stream?language=${language}&task=${task}`);
                
                socket.onopen = () => {
                    statusIndicator.className = 'status-indicator status-connected';
                    statusText.textContent = 'Connected';
                    startBtn.disabled = false;
                };
                
                socket.onmessage = (event) => {
                    try {
                        const response = JSON.parse(event.data);
                        
                        if (response.type === 'transcription' && response.text) {
                            transcript.innerHTML += response.text + ' ';
                        } else if (response.type === 'connection_established') {
                            console.log('Connection established:', response);
                        }
                    } catch (error) {
                        console.error('Error parsing message:', error);
                        transcript.innerHTML += event.data + ' ';
                    }
                };
                
                socket.onclose = () => {
                    statusIndicator.className = 'status-indicator status-disconnected';
                    statusText.textContent = 'Disconnected';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    
                    if (isRecording) {
                        stopRecording();
                    }
                };
                
                socket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    statusIndicator.className = 'status-indicator status-disconnected';
                    statusText.textContent = 'Error: Could not connect';
                };
            }
            
            // Start recording
            function startRecording() {
                if (isRecording) return;
                
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then((stream) => {
                        audioStream = stream;
                        setupVisualizer(stream);
                        
                        // Create MediaRecorder
                        mediaRecorder = new MediaRecorder(stream);
                        
                        mediaRecorder.ondataavailable = (event) => {
                            if (socket && socket.readyState === WebSocket.OPEN) {
                                socket.send(event.data);
                            }
                        };
                        
                        // Start recording
                        mediaRecorder.start(250); // Send data every 250ms
                        isRecording = true;
                        
                        // Update buttons
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                    })
                    .catch((error) => {
                        console.error('Error accessing microphone:', error);
                        alert('Error accessing microphone. Please check permissions.');
                    });
            }
            
            // Stop recording
            function stopRecording() {
                if (!isRecording) return;
                
                // Stop MediaRecorder
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                
                // Stop audio tracks
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                }
                
                // Cancel visualization
                if (drawVisual) {
                    cancelAnimationFrame(drawVisual);
                }
                
                isRecording = false;
                
                // Update buttons
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
            
            // Upload file
            function uploadFile() {
                const file = audioFileInput.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('language', languageSelect.value);
                formData.append('task', taskSelect.value);
                
                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Transcribing...';
                
                fetch('/api/transcribe', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        transcript.innerHTML += '\\n[File Upload] ' + data.transcription + '\\n';
                    } else {
                        console.error('Error:', data);
                        alert('Error transcribing file: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => {
                    console.error('Error uploading file:', error);
                    alert('Error uploading file: ' + error.message);
                })
                .finally(() => {
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = 'Transcribe File';
                });
            }
            
            // Event Listeners
            startBtn.addEventListener('click', () => {
                if (!socket || socket.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                    setTimeout(startRecording, 1000); // Wait for connection
                } else {
                    startRecording();
                }
            });
            
            stopBtn.addEventListener('click', stopRecording);
            
            audioFileInput.addEventListener('change', () => {
                if (audioFileInput.files.length > 0) {
                    fileName.textContent = audioFileInput.files[0].name;
                    uploadBtn.disabled = false;
                } else {
                    fileName.textContent = 'No file chosen';
                    uploadBtn.disabled = true;
                }
            });
            
            uploadBtn.addEventListener('click', uploadFile);
            
            languageSelect.addEventListener('change', () => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ language: languageSelect.value }));
                }
            });
            
            taskSelect.addEventListener('change', () => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ task: taskSelect.value }));
                }
            });
            
            // Initialize connection on page load
            window.addEventListener('load', () => {
                // Connect when the user interacts with the page
                document.body.addEventListener('click', () => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) {
                        connectWebSocket();
                    }
                }, { once: true });
            });
        </script>
    </body>
    </html>
    """)

# Model configuration
MODEL_CONFIG = {
    "model_id": "Saintdannyyy/kasayie-whisper-small-akan-nonstandard",
    "model_name": "KasaYie ASR Model",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "language_map": {
        "ak": "yo",     # We use Yoruba as the language tag for Akan/Twi
        "tw": "yo",     # Alternative code for Twi
        "en": "en",     # English
        "yo": "yo"      # Yoruba (for compatibility)
    },
    "default_language": "ak",
    "default_task": "transcribe",
    "sampling_rate": 16000,
    "chunk_duration": 2.0,      # Duration in seconds for streaming chunks
    "overlap_duration": 0.5,    # Overlap between chunks in seconds
    "temperature": 0.2,         # Sampling temperature for generation
    "silence_threshold": 0.01,  # Threshold for silence detection
    "silence_duration": 0.5     # Duration in seconds to consider silence
}

# Initialize app state
app.state.start_time = time.time()

# Model cache with lock for thread safety
model_lock = threading.Lock()
model_cache = {
    "processor": None,
    "model": None,
    "last_loaded": None
}

# Stats for monitoring
stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_processing_time": 0,
    "total_audio_duration": 0
}
stats_lock = threading.Lock()

# Pydantic models for requests and responses
class TranscriptionRequest(BaseModel):
    language: str = Field(MODEL_CONFIG["default_language"], description="Language code (ak for Akan/Twi, en for English)")
    task: str = Field(MODEL_CONFIG["default_task"], description="Task type: 'transcribe' or 'translate'")

class TranscriptionResponse(BaseModel):
    transcription: str
    status: str
    processing_time: float
    language: str
    confidence: Optional[float] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model_id: str
    device: str
    uptime: float
    total_requests: int
    avg_processing_time: float

# Load model and processor
def load_model():
    global model_cache
    
    with model_lock:
        # If model is already loaded and recent (within last 2 hours), reuse it
        if (model_cache["model"] is not None and 
            model_cache["last_loaded"] is not None and 
            time.time() - model_cache["last_loaded"] < 7200):
            logger.info("Using cached model")
            return model_cache["processor"], model_cache["model"]
        
        logger.info(f"Loading model from {MODEL_CONFIG['model_id']}...")
        try:
            processor = WhisperProcessor.from_pretrained(MODEL_CONFIG['model_id'])
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_id']).to(MODEL_CONFIG['device'])
            
            # Apply half-precision for GPU if available for better performance
            if MODEL_CONFIG['device'] == 'cuda':
                model = model.half()  # Use FP16 for faster inference on GPU
            
            model_cache["processor"] = processor
            model_cache["model"] = model
            model_cache["last_loaded"] = time.time()
            
            logger.info(f"Model loaded successfully on {MODEL_CONFIG['device']}")
            return processor, model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

# Initialize Whisper pipeline
whisper_pipeline = pipeline(
    "automatic-speech-recognition", 
    model=MODEL_CONFIG['model_id'],
    device=MODEL_CONFIG['device']
)

# Background task to pre-load model
@app.on_event("startup")
async def startup_event():
    logger.info("Starting KasaYie ASR server...")
    
    # Start a background thread to load the model
    threading.Thread(target=load_model, daemon=True).start()
    logger.info("Model loading started in background thread")

# Audio processing helper functions
def detect_silence(audio_data, threshold=MODEL_CONFIG["silence_threshold"], 
                   min_silence_duration=MODEL_CONFIG["silence_duration"], 
                   sample_rate=MODEL_CONFIG["sampling_rate"]):
    """Detect silence in audio data"""
    # Calculate energy
    energy = np.abs(audio_data)
    
    # Find segments below threshold
    is_silence = energy < threshold
    
    # Count consecutive silence frames
    silence_frames = int(min_silence_duration * sample_rate)
    
    # If we have enough consecutive silence frames, return True
    for i in range(len(is_silence) - silence_frames + 1):
        if np.all(is_silence[i:i+silence_frames]):
            return True
    
    return False

def process_audio(audio_data, sr, processor, model, language="ak", task="transcribe"):
    """Process audio data and return transcription"""
    start_time = time.time()
    
    try:
        # Update stats
        with stats_lock:
            stats["total_requests"] += 1
        
        # Skip processing if audio is silent
        if detect_silence(audio_data):
            logger.info("Silent audio detected, skipping processing")
            return {
                "transcription": "",
                "status": "silent",
                "processing_time": time.time() - start_time,
                "language": language
            }
        
        # Resample to model sampling rate if needed
        if sr != MODEL_CONFIG["sampling_rate"]:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=MODEL_CONFIG["sampling_rate"])
            sr = MODEL_CONFIG["sampling_rate"]
        
        # Apply pre-emphasis filter to enhance speech signal (especially for impaired speech)
        audio_data = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])
        
        # Normalize audio to improve consistency
        audio_data = librosa.util.normalize(audio_data)
        
        # Use the language mapping
        whisper_lang = MODEL_CONFIG["language_map"].get(language, "yo")
        
        # Generate transcription using pipeline
        result = whisper_pipeline(
            audio_data, 
            language=whisper_lang,
            task=task,
            chunk_length_s=MODEL_CONFIG["chunk_duration"],
            return_timestamps=True
        )
        
        transcription = result.get("text", "")
        processing_time = time.time() - start_time
        
        # Update stats
        with stats_lock:
            stats["successful_requests"] += 1
            stats["avg_processing_time"] = ((stats["avg_processing_time"] * (stats["successful_requests"] - 1)) + 
                                            processing_time) / stats["successful_requests"]
            stats["total_audio_duration"] += len(audio_data) / sr
        
        logger.info(f"Transcription completed in {processing_time:.2f}s: {transcription}")
        
        return {
            "transcription": transcription,
            "status": "success",
            "processing_time": processing_time,
            "language": language,
            "confidence": 0.85  # Placeholder for confidence score
        }
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        
        # Update stats
        with stats_lock:
            stats["failed_requests"] += 1
        
        processing_time = time.time() - start_time
        return {
            "transcription": "",
            "status": "error",
            "error": str(e),
            "processing_time": processing_time,
            "language": language
        }

# Validate language and task parameters
def validate_params(
    language: str = Query(MODEL_CONFIG["default_language"]), 
    task: str = Query(MODEL_CONFIG["default_task"])
) -> Dict[str, str]:
    if language not in MODEL_CONFIG["language_map"]:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{language}'")
    if task not in ["transcribe", "translate"]:
        raise HTTPException(status_code=400, detail=f"Invalid task '{task}'")
    return {"language": language, "task": task}

# POST /api/transcribe endpoint
@app.post("/api/transcribe", response_model=TranscriptionResponse, responses={400: {"model": ErrorResponse}})
async def transcribe_audio(
    file: UploadFile = File(...), 
    params: Dict[str, str] = Depends(validate_params)
):
    try:
        audio_bytes = await file.read()
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        
        processor, model = load_model()
        result = process_audio(audio_data, sr, processor, model, params["language"], params["task"])
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket /api/stream
@app.websocket("/api/stream")
async def stream_audio(websocket: WebSocket, language: str = Query("ak"), task: str = Query("transcribe")):
    await websocket.accept()
    await websocket.send_json({"type": "connection_established", "message": "WebSocket connected."})

    try:
        buffer = bytes()
        processor, model = load_model()

        while True:
            message = await websocket.receive_bytes()
            buffer += message

            if len(buffer) >= int(MODEL_CONFIG["sampling_rate"] * MODEL_CONFIG["chunk_duration"] * 2):  # 16-bit PCM = 2 bytes/sample
                audio_np, _ = sf.read(io.BytesIO(buffer), dtype='float32')
                result = process_audio(audio_np, MODEL_CONFIG["sampling_rate"], processor, model, language, task)
                if result["status"] == "success":
                    await websocket.send_json({"type": "transcription", "text": result["transcription"]})
                buffer = bytes()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close(code=1011)

# GET /health endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - app.state.start_time
    return {
        "status": "ok",
        "model_id": MODEL_CONFIG["model_id"],
        "device": MODEL_CONFIG["device"],
        "uptime": uptime,
        "total_requests": stats["total_requests"],
        "avg_processing_time": stats["avg_processing_time"]
    }

# get /stats endpoint
@app.get("/stats")
def get_stats():
    return stats

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    # To run the app, use the command: uvicorn app:app --host 0.0.0.0 --port 8000