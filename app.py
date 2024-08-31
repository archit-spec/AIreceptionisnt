import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ai_receptionist import AIReceptionist
from vector_db import VectorDB
#from fastapi_cache import FastAPICache
#from fastapi_cache.decorator import cache

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
print("Available collections:", vector_db.get_collections())

# Initialize the collection and load data only if it doesn't exist
if "emergency_instructions" not in [c.name for c in vector_db.get_collections().collections]:
    vector_db.initialize_collection()
    vector_db.load_data('emergency_instructions.json')
else:
    print("Collection 'emergency_instructions' already exists. Skipping initialization.")

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
                response = await receptionist.process_input(data)
                await websocket.send_text(response)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                break
            except Exception as e:
                print(f"An error occurred in websocket_endpoint: {str(e)}")
                await websocket.send_text("I'm sorry, an error occurred. Please try again.")
    finally:
        print("WebSocket connection closed")

@app.get("/state")
async def get_state(request: Request):
    state_context = receptionist.get_state_context()
    return templates.TemplateResponse("state.html", {"request": request, "state_context": state_context})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
