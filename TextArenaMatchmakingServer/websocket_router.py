import time, json, logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from starlette.websockets import WebSocketState

from utils import (
    supabase,
    active_connections,
    general_information,
    matchmaking_registry,
    join_matchmaking,
    leave_matchmaking,
)
from config import HUMANITY_MODEL_ID




router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Attempt to identify user
    token = websocket.cookies.get("user_id") or websocket.query_params.get("user_id")
    if token:
        # It's a human
        human_result = supabase.table("humans").select("id").eq("cookie_id", token).execute()
        if not human_result.data:
            await websocket.send_text("Error: Human not found.")
            await websocket.close()
            return
        human_id = human_result.data[0]["id"]
        model_id = HUMANITY_MODEL_ID
        model_name = "Humanity"
        logging.info(f"Human connected with id {human_id}")

        
    else:
        # Possibly a submitted model
        query_params = websocket.query_params
        model_name = query_params.get("model_name")
        token = query_params.get("model_token")
        if not model_name or not token:
            await websocket.send_text("Error: Missing model credentials.")
            await websocket.close()
            return
        # Validate
        model_result = (
            supabase.table("models")
            .select("id")
            .eq("model_name", model_name)
            .eq("model_token", token)
            .execute()
        )
        if not model_result.data:
            await websocket.send_text("Error: Invalid model credentials.")
            await websocket.close()
            return
        human_id = None
        model_id = model_result.data[0]["id"]
        logging.info(f"Model connected with id={model_id}, human_id={human_id}, name={model_name}")

    # Store in active_connections
    active_connections[token] = {
        "model_name": model_name,
        "model_id": model_id,
        "human_id": human_id,
        "ws": websocket,
    }
    logging.info(f"Stored connection for {model_name}, model_id={model_id}, human_id={human_id}, token={token}")

    try:
        while True:
            data = await websocket.receive_text()
            logging.info(f"Received from {model_name}, model_id={model_id}, human_id={human_id}: {data}")
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text("Error: Invalid JSON payload.")
                continue

            command = payload.get("command")

            if command == "queue":
                await command_queue(payload, token, websocket)
            elif command == "leave":
                await command_leave(token, websocket)
            else:
                await websocket.send_text(json.dumps({"command": "error", "message": f"Unknown command: {command}"}))

    except WebSocketDisconnect:
        logging.info(f"{model_name}, model_id={model_id}, human_id={human_id}, token={token} disconnected.")
        ## Cleanup
        # remove from active_connections
        leave_matchmaking(token)
        if token in active_connections:
            del active_connections[token]


async def command_queue(payload: Dict, token: str, websocket: WebSocket):
    env_ids = payload.get("environments")
    if not isinstance(env_ids, list):
        await websocket.send_text("Error: 'environments' must be a list.")
        return

    # Join matchmaking
    join_matchmaking(
        token=token,
        env_ids=env_ids,
        model_id=active_connections[token]["model_id"],
        human_id=active_connections[token]["human_id"],
    )

    response_payload = {
        "command": "queued",
        "avg_queue_time": float(
            np.mean(general_information["past_queue_times"])
        ) if general_information["past_queue_times"] else 69,
        "num_players_in_queue": len(active_connections)
    }
    await websocket.send_text(json.dumps(response_payload))


async def command_leave(token: str, websocket: WebSocket):
    global active_connections
    leave_matchmaking(token)
    if token in active_connections:
        del active_connections[token]
    await websocket.send_text(json.dumps({"command": "left"}))