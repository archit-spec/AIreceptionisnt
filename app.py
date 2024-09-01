import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ai_receptionist import AIReceptionist
from vector_db import VectorDB
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
    
#@cache()
#async def get_cache():
#    return 1
# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

receptionist = AIReceptionist()

vector_db = VectorDB()
logger.info("Available collections: %s", vector_db.get_collections())

# Initialize the collection and load data only if it doesn't exist
if "emergency_instructions" not in [c.name for c in vector_db.get_collections().collections]:
    vector_db.initialize_collection()
    vector_db.load_data('emergency_instructions.json')
    logger.info("Initialized 'emergency_instructions' collection and loaded data.")
else:
    logger.info("Collection 'emergency_instructions' already exists. Skipping initialization.")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_text()
                logger.info("Received message: %s", data)
                response = await receptionist.process_input(data)
                logger.info("Sending response: %s", response)
                await websocket.send_text(response)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error("An error occurred in websocket_endpoint: %s", str(e))
                await websocket.send_text("I'm sorry, an error occurred. Please try again.")
    finally:
        logger.info("WebSocket connection closed")

@app.get("/state")
async def get_state(request: Request):
    state_context = receptionist.get_state_context()
    return templates.TemplateResponse("state.html", {"request": request, "state_context": state_context})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
