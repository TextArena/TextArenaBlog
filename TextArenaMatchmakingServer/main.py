# main.py
import uvicorn, asyncio 
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response  # Add this missing import

from api import router as api_router
from websocket_router import router as ws_router
from utils import matchmaking_loop, replenish_server_pool, periodic_game_server_health_check, logging
from config import NUM_CONNECTION_LIMIT



def get_allowed_origins():
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://www.textarena.ai",
        "https://textarena.ai",
        "https://api.textarena.ai",
        "https://matchmaking.textarena.ai",
    ]

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-User-Token"],
    expose_headers=["*"],
    max_age=3600
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    response = Response()
    origin = request.headers.get("origin", "")
    if origin in get_allowed_origins():
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-User-Token"
        response.headers["Access-Control-Max-Age"] = "3600"
    return response


# Custom CORS handling if you want to dynamically set credentials
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin", "")
    
    # Log CORS details for debugging
    logging.debug(f"Request from origin: {origin}")
    logging.debug(f"Method: {request.method}")
    logging.debug(f"Path: {request.url.path}")
    
    # Special handling for OPTIONS requests
    if request.method == "OPTIONS":
        allowed_origins = get_allowed_origins()
        if origin in allowed_origins:
            # For OPTIONS, return an immediate response
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-User-Token"
            response.headers["Access-Control-Max-Age"] = "3600"
            return response
    
    # For all other requests, proceed with normal handling
    response = await call_next(request)
    
    # Add CORS headers to the response
    if origin in get_allowed_origins():
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

# Include routers
app.include_router(api_router)
app.include_router(ws_router)

@app.on_event("startup")
async def on_startup():
    logging.info("Starting background tasks...")

    asyncio.create_task(matchmaking_loop())
    asyncio.create_task(replenish_server_pool())
    asyncio.create_task(periodic_game_server_health_check())

    logging.info("All background tasks have been launched.")

@app.get("/")
def root():
    return {"message": "Matchmaking server is alive"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)