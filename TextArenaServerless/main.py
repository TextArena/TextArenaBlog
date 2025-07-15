import sys, json, time, random
import uvicorn, asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client
from typing import List, Dict
import textarena as ta
import os
import threading
from fastapi import Path, Depends
from utils import get_participant_label_from_token

import logging
import traceback

# Set up logging with timestamps
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Local imports
from utils import update_trueskill_scores, determine_outcomes, update_game_state


from pydantic import BaseModel

class InitializeRequest(BaseModel):
    environment_id: int
    env_id: str
    tokens: List[str]

# constants
INITIAL_CONNECTION_TIMEOUT = 180

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Global flag to track server shutdown request
server_shutdown_requested = False

# FastAPI app with CORS settings
def get_allowed_origins():
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://www.textarena.ai",
        "https://textarena.ai",
        "https://api.textarena.ai",
        "https://matchmaking.textarena.ai",
        "https://gamehost.textarena.ai",
        ""
    ]

def force_exit_after_delay(delay=15):
    """Force exit the process after a delay, as a fallback."""
    def _exit_func():
        time.sleep(delay)
        logging.warning(f"Forcing exit after {delay} seconds")
        os._exit(0)  # Hard exit that can't be caught or blocked
    
    # Start a daemon thread that will force exit
    exit_thread = threading.Thread(target=_exit_func)
    exit_thread.daemon = True
    exit_thread.start()
    logging.info(f"Scheduled forced exit in {delay} seconds")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # allow_origins=get_allowed_origins(),
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Add this middleware to check for shutdown requests
@app.middleware("http")
async def check_shutdown(request: Request, call_next):
    if server_shutdown_requested:
        # Return a 503 Service Unavailable during shutdown
        return JSONResponse(
            status_code=503,
            content={"detail": "Server is shutting down."}
        )
    return await call_next(request)


class Game:
    def __init__(self, environment_id: int, env_id: str, tokens: List[str], supabase: Client):
        self.initialization_ts = time.time()
        self.game_active = False 

        # Add shutdown tracking
        self.shutdown_initiated = False

        # Add message tracking
        self.game_over_messages_sent = {pid: False for pid in range(len(tokens))}
        self.all_messages_sent = False

        self.environment_id = environment_id
        self.env_id = env_id
        self.tokens = tokens
        self.supabase = supabase

        self.player_dict = {}
        self.token_to_pid = {}
        player_ids = list(range(len(tokens)))
        random.shuffle(player_ids)
        for token, player_id in zip(tokens, player_ids):
            self.player_dict[player_id] = {"token": token, "websocket": None}
            self.token_to_pid[token] = player_id

        self.connected = {pid: True for pid in self.player_dict}
        self.done = False
        self.step_info = {}
        self.game_info = {}
        self.current_player_id = None
        self.moves = {token: [] for token in tokens}
        self.move_deadline = None  # Deadline current player's move

        self.env = ta.make(env_id=env_id)
        ## check if environment is already wrapped
        if not isinstance(self.env, ta.wrappers.ActionFormattingWrapper):
            self.env = ta.wrappers.ActionFormattingWrapper(env=self.env)
        self.env.reset(num_players=len(self.tokens))

        # Start background timeout checker
        asyncio.create_task(self._check_timeouts())

    def is_valid_token(self, token: str) -> bool:
        return token in self.tokens

    def add_websocket(self, token: str, websocket: WebSocket):
        pid = self.token_to_pid[token]
        self.player_dict[pid]["websocket"] = websocket
        if all(self.player_dict[pid]["websocket"] is not None for pid in self.player_dict):
            asyncio.create_task(self._get_and_send_observation())
            self.game_active = True

    def check_turn(self, token: str) -> bool:
        return self.env.state.current_player_id == self.token_to_pid[token]

    def step_env(self, token: str, action: str):
        player_pid = self.token_to_pid.get(token, "unknown")
        logging.info(f"Processing environment step for Player {player_pid}")
        
        if not self.done:
            # Process the action
            try:
                self.done, self.step_info = self.env.step(action=action)
                logging.info(f"Environment step completed. Done: {self.done}, Info: {self.step_info}")
                
                self.moves[token][-1]["action"] = action
                self.moves[token][-1]["timestamp_action"] = time.time()
                logging.info(f"Action recorded for Player {player_pid}")
            except Exception as e:
                logging.error(f"Error during environment step: {e}")
                self.done = True
                self.step_info = {"error": f"Environment step error: {str(e)}"}

        if not self.done:
            # Get and send observation for the next player's turn
            logging.info("Game continues, getting next observation")
            asyncio.create_task(self._get_and_send_observation())
        else:
            # Game is over, but don't close connections yet
            rewards, game_info = self.env.close()
            logging.info(f"Game is over. Reason: {game_info[0].get("reason", "unknown")}")
            
            # Now update the game state and send final messages
            asyncio.create_task(update_game_state(
                game_obj=self,
                rewards=rewards,
                reason=game_info[0].get("reason", "unknown"),
                supabase=self.supabase
            ))

            # Schedule shutdown AFTER messages are confirmed as sent
            asyncio.create_task(self._wait_and_shutdown(10))  # 10 second deadline as a safety measure

    async def _wait_and_shutdown(self, max_delay=10):
        """Wait for all messages to be sent before shutting down with a safety timeout."""
        start_time = time.time()
        while not self.all_messages_sent:
            # Check if we've waited too long
            if time.time() - start_time > max_delay:
                logging.warning(f"Timeout waiting for all game over messages. Proceeding with shutdown.")
                break
                
            await asyncio.sleep(0.5)  # Check frequently
            # Refresh the check in case messages were sent while waiting
            self.all_messages_sent = all(self.game_over_messages_sent.values())
        
        # Now proceed with delayed shutdown
        await self._delayed_shutdown(5)  # 5 second additional delay for messages to be processed

    async def _delayed_shutdown(self, delay=5):
        """Shutdown with a delay to allow final messages to be sent."""
        try:
            # Wait to ensure all messages are sent
            await asyncio.sleep(delay)
            
            # Track which websockets we've already closed
            closed_websockets = set()
            
            # Send a final notification
            for pid in self.player_dict:
                websocket = self.player_dict[pid]["websocket"]
                if websocket and websocket not in closed_websockets and not getattr(websocket, 'closed', True):
                    try:
                        message = {
                            "command": "server_shutdown",
                            "message": "Game server is shutting down"
                        }
                        await websocket.send_text(json.dumps(message))
                        logging.info(f"Shutdown message sent to Player {pid}")
                        
                        # Close the websocket gracefully and mark it as closed
                        await websocket.close(code=1000)
                        closed_websockets.add(websocket)
                        logging.info(f"Closed websocket for Player {pid}")
                    except Exception as e:
                        logging.error(f"Error during shutdown for Player {pid}: {e}")
            
            # Final wait before shutdown
            await asyncio.sleep(2)
            
            # Mark as ready for final shutdown
            self.shutdown_initiated = True
            
            # Schedule the actual shutdown task
            asyncio.create_task(self._final_shutdown())
        except Exception as e:
            logging.error(f"Error in delayed shutdown: {e}")
            # Still try to shut down even if there was an error
            asyncio.create_task(self._final_shutdown())

    async def _final_shutdown(self):
        """Final shutdown process that exits the Fargate task."""
        try:
            # Last wait to ensure all messages are processed
            await asyncio.sleep(3)
            
            logging.info("Executing final Fargate task shutdown")
            
            # Set up a fallback forced exit in case the normal exit gets stuck
            force_exit_after_delay(15)  # Force exit after 15 seconds if normal exit fails
            
            # Try normal exit first
            try:
                logging.info("Shutting down Fargate task.")
                sys.exit(0)  # This might raise an exception that gets caught
            except SystemExit:
                # This is expected, but the exception might be caught by uvicorn
                # The force_exit fallback will handle this case
                pass
            except Exception as e:
                logging.error(f"Error during sys.exit: {e}")
                # Fall back to os._exit
                os._exit(0)
        except Exception as e:
            logging.error(f"Error during final shutdown: {e}")
            # Force exit even if there's an error
            os._exit(1)

    async def _get_and_send_observation(self):
        logging.info("Getting next observation")
        try:
            # Get the player ID and raw observation tuples from the environment state
            next_player_id = self.env.state.current_player_id
            raw_observation_tuples = self.env.state.get_current_player_observation()
            
            logging.info(f"Raw observation received for Player {next_player_id}, length: {len(raw_observation_tuples) if raw_observation_tuples else 0}")
            
            self.current_player_id = next_player_id
            token = self.player_dict[next_player_id]["token"]
            
            # Store the raw observation tuples for moves tracking
            self.moves[token].append({
                "observation": raw_observation_tuples,  # Store the raw tuples
                "timestamp_observation": time.time(),
                "action": None,
                "timestamp_action": None,
            })
            logging.debug(f"Move recorded for Player {next_player_id}")
            
            # Set move deadline (180 seconds from now)
            self.move_deadline = time.time() + 180
            next_player_ws = self.player_dict[next_player_id]["websocket"]
            
            # Convert the observation tuples to JSON-serializable format
            # Each tuple is (sender_id, message, ObservationType)
            serializable_observation = []
            if raw_observation_tuples:
                for sender_id, message, obs_type in raw_observation_tuples:
                    serializable_observation.append([
                        sender_id, 
                        message, 
                        obs_type.value  # Convert ObservationType enum to int for JSON
                    ])
            
            message = {
                "command": "observation", 
                "observation": serializable_observation,  # Send as list of [sender_id, message, obs_type_int]
                "player_id": self.current_player_id
            }
            
            logging.debug(f"Sending observation to Player {next_player_id}: {len(serializable_observation)} observation tuples")
            
            try:
                await next_player_ws.send_text(json.dumps(message))
                logging.info(f"Observation sent to Player {next_player_id}")
            except Exception as e:
                logging.error(f"Error sending observation to Player {next_player_id}: {e}")
                logging.error(traceback.format_exc())
                self.connected[next_player_id] = False
                await self._send_timeout_messages(timeout_pid=next_player_id)
        except Exception as e:
            logging.error(f"Error in _get_and_send_observation: {e}")
            logging.error(traceback.format_exc())

    async def _check_timeouts(self):
        while not self.done:
            await asyncio.sleep(5)  # Check every 5 seconds
            if (
                self.current_player_id is not None and 
                self.move_deadline is not None and 
                self.move_deadline < time.time()
            ):
                self.connected[self.current_player_id] = False
                await self._send_timeout_messages(timeout_pid=self.current_player_id)

            if self.game_active==False and time.time()-self.initialization_ts > INITIAL_CONNECTION_TIMEOUT:
                # somebody didn't connect on time. time everybody out.
                for pid in self.player_dict:
                    websocket = self.player_dict[pid]["websocket"]
                    if websocket:
                        message = {"command": "timed_out", "message": f"Some players did not connect to the server on time."}
                        await websocket.send_text(json.dumps(message))

    async def shutdown_task(self):
        """Handle cleanup before shutdown, but defer to _final_shutdown for the actual exit."""
        try:
            # Only proceed if we haven't already initiated shutdown
            if hasattr(self, 'shutdown_initiated') and self.shutdown_initiated:
                logging.info("Shutdown already initiated, skipping redundant shutdown")
                return
                
            # Check if any websockets are still open
            open_count = 0
            for pid in self.player_dict:
                websocket = self.player_dict[pid]["websocket"]
                if websocket and not getattr(websocket, 'closed', True):
                    open_count += 1
                    try:
                        message = {
                            "command": "server_shutdown",
                            "message": "Game server is shutting down"
                        }
                        await websocket.send_text(json.dumps(message))
                        await websocket.close(code=1000)
                        logging.info(f"Closed websocket for Player {pid} during shutdown_task")
                    except Exception as e:
                        # Just log the error and continue
                        logging.warning(f"Error closing websocket for player {pid}: {e}")
            
            logging.info(f"Shutdown task completed, closed {open_count} open websockets")
            
            # Mark as initiated and schedule the final shutdown
            self.shutdown_initiated = True
            asyncio.create_task(self._final_shutdown())
        except Exception as e:
            logging.error(f"Error during shutdown task: {e}")
            # Still try to exit
            asyncio.create_task(self._final_shutdown())


    async def _send_timeout_messages(self, timeout_pid: int):
        # set done
        self.done = True 
        rewards = {
            pid: -1 if pid == timeout_pid else 0
            for pid in self.player_dict.keys()
        }

        # Track if we've sent messages to all players
        for pid in self.player_dict:
            self.game_over_messages_sent[pid] = False

        # First send the timeout messages
        for pid in self.player_dict:
            websocket = self.player_dict[pid]["websocket"]
            if websocket:
                try:
                    timed_out_token = self.player_dict[timeout_pid]["token"]
                    participant_label = get_participant_label_from_token(timed_out_token, self.supabase)

                    message = {
                        "command": "timed_out",
                        "message": f"{participant_label} (Player {timeout_pid}) timed out. Game complete."
                    }

                    await websocket.send_text(json.dumps(message))
                    self.game_over_messages_sent[pid] = True
                    logging.info(f"Timeout message sent to Player {pid}")
                except Exception as e:
                    logging.error(f"Error sending timeout message to Player {pid}: {e}")
            else:
                # Mark as sent if websocket doesn't exist
                self.game_over_messages_sent[pid] = True

        # Check if all messages have been sent
        self.all_messages_sent = all(self.game_over_messages_sent.values())
        
        # Now update game state
        # await update_game_state(self, rewards, "Timed out", self.supabase)
        
        # Wait and initiate shutdown only after confirming messages are sent
        asyncio.create_task(self._wait_and_shutdown(10))


    async def set_timeout(self, token:str):
        self.connected[self.token_to_pid[token]] = False
        await self._send_timeout_messages(timeout_pid=self.token_to_pid[token])

game_obj = None

@app.post("/initialize")
async def initialize_server(request: InitializeRequest):
    global game_obj
    game_obj = Game(
        environment_id=request.environment_id, 
        env_id=request.env_id, 
        tokens=request.tokens,
        supabase=supabase
    )
    return {"message": "Game initialized"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Server is running."}

# Add this function to extract game_id from path parameters
@app.get("/game/{game_id}/health")
async def game_health_check(game_id: str = Path(...)):
    """Health check endpoint accessible through the ALB routing pattern"""
    return {"status": "ok", "message": "Server is running.", "game_id": game_id}


@app.websocket("/game/{game_id}/ws")
async def game_websocket_endpoint(websocket: WebSocket, game_id: str):
    logging.info(f"New WebSocket connection request for game {game_id}")
    await websocket.accept()
    logging.info(f"WebSocket connection accepted for game {game_id}")
    
    token = websocket.query_params.get("token")
    logging.info(f"Connection with token: {token} for game {game_id}")
    
    # Set up some variables to track state
    connection_active = True
    player_pid = None

    if game_obj is None:
        logging.error(f"Game not initialized for game {game_id}")
        await websocket.send_text(json.dumps({"command": "error", "message": "Game not initialized"}))
        await websocket.close(code=1000)
        return

    if token is None or not game_obj.is_valid_token(token):
        logging.error(f"Invalid token: {token}")
        await websocket.send_text(json.dumps({"command": "error", "message": "Invalid token"}))
        await websocket.close(code=1000)
        return

    # Get player ID for logging
    if token in game_obj.token_to_pid:
        player_pid = game_obj.token_to_pid[token]
        logging.info(f"Player {player_pid} connected with token {token}")

    # Add the websocket to the game
    try:
        logging.info(f"Adding websocket for Player {player_pid} to game")
        game_obj.add_websocket(token=token, websocket=websocket)
        logging.info(f"Websocket added for Player {player_pid}")
    except Exception as e:
        logging.error(f"Error adding websocket for Player {player_pid}: {e}")
        logging.error(traceback.format_exc())
        await websocket.close(code=1011)  # Internal error
        return

    try:
        # Keep the connection alive
        while connection_active and not (game_obj.done and game_obj.game_over_messages_sent.get(player_pid, False)):
            try:
                logging.debug(f"Waiting for message from Player {player_pid}")
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)  # 30-second timeout
                logging.info(f"Received data from Player {player_pid}: {data[:100]}...")
                
                try:
                    payload = json.loads(data)
                    command = payload.get("command")
                    logging.info(f"Player {player_pid} sent command: {command}")

                    if command == "leave":
                        logging.info(f"Player {player_pid} requested to leave")
                        await game_obj.set_timeout(token)
                        connection_active = False
                    elif command == "action":
                        logging.info(f"Player {player_pid} sent action: {payload.get('action', '')[:50]}...")
                        
                        # First send acknowledgment before processing the action
                        try:
                            ack_msg = json.dumps({
                                "command": "action_ack",
                                "message": "Action received"
                            })
                            logging.debug(f"Sending action acknowledgment to Player {player_pid}")
                            await websocket.send_text(ack_msg)
                            logging.info(f"Action acknowledgment sent to Player {player_pid}")
                        except Exception as e:
                            logging.error(f"Error sending acknowledgment to Player {player_pid}: {e}")
                            logging.error(traceback.format_exc())
                            raise
                        
                        # Then process the action
                        logging.debug(f"Processing action from Player {player_pid}")
                        await command_action(payload, token, websocket)
                        logging.info(f"Action processed for Player {player_pid}")
                        
                    elif command == "ping":
                        # Respond to ping with pong
                        logging.debug(f"Received ping from Player {player_pid}")
                        await websocket.send_text(json.dumps({"command": "pong"}))
                        logging.debug(f"Sent pong to Player {player_pid}")
                    elif command == "pong":
                        # This is a response to our ping, just acknowledge it
                        logging.debug(f"Received pong from Player {player_pid}")
                        # No response needed
                    else:
                        logging.warning(f"Unknown command from Player {player_pid}: {command}")
                        await websocket.send_text(json.dumps({"command": "error", "message": f"Unknown command: {command}"}))
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON from Player {player_pid}: {e}")
                    logging.error(f"Raw data: {data}")
                    await websocket.send_text(json.dumps({"command": "error", "message": "Invalid JSON payload"}))
                except Exception as e:
                    logging.error(f"Error processing command from Player {player_pid}: {e}")
                    logging.error(traceback.format_exc())
                    await websocket.send_text(json.dumps({"command": "error", "message": f"Command processing error: {str(e)}"}))
                    
            except asyncio.TimeoutError:
                # Send a ping to keep the connection alive
                logging.debug(f"No message received from Player {player_pid} in 30s, sending ping")
                try:
                    await websocket.send_text(json.dumps({"command": "ping"}))
                    logging.debug(f"Ping sent to Player {player_pid}")
                except Exception as e:
                    logging.error(f"Error sending ping to Player {player_pid}: {e}")
                    logging.error(traceback.format_exc())
                    connection_active = False
            
            # Check if the game is done and this player has received their game over message
            if game_obj.done and game_obj.game_over_messages_sent.get(player_pid, False):
                logging.info(f"Game is done and Player {player_pid} has received game over message. Breaking loop.")
                break
                
    except WebSocketDisconnect as e:
        logging.warning(f"WebSocket disconnected for Player {player_pid}, code: {e.code}")
        # Don't set timeout immediately if game is already done and we're just waiting for messages
        if not game_obj.done:
            await game_obj.set_timeout(token)
    except Exception as e:
        logging.error(f"Unexpected error for Player {player_pid}: {e}")
        logging.error(traceback.format_exc())
        try:
            await websocket.send_text(json.dumps({"command": "error", "message": f"Server error: {str(e)}"}))
        except:
            logging.error("Could not send error message to client")
    
    # Final cleanup
    logging.info(f"WebSocket handler ending for Player {player_pid}")
    
    # Only set timeout if the game isn't already done
    if not game_obj.done:
        try:
            logging.info(f"Setting timeout for Player {player_pid}")
            await game_obj.set_timeout(token)
            logging.info(f"Timeout set for Player {player_pid}")
        except Exception as e:
            logging.error(f"Error during timeout handling: {e}")
            logging.error(traceback.format_exc())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logging.info("New WebSocket connection request")
    await websocket.accept()
    logging.info("WebSocket connection accepted")
    
    token = websocket.query_params.get("token")
    logging.info(f"Connection with token: {token}")
    
    # Set up some variables to track state
    connection_active = True
    player_pid = None

    if game_obj is None:
        logging.error("Game not initialized")
        await websocket.send_text(json.dumps({"command": "error", "message": "Game not initialized"}))
        await websocket.close(code=1000)
        return

    if token is None or not game_obj.is_valid_token(token):
        logging.error(f"Invalid token: {token}")
        await websocket.send_text(json.dumps({"command": "error", "message": "Invalid token"}))
        await websocket.close(code=1000)
        return

    # Get player ID for logging
    if token in game_obj.token_to_pid:
        player_pid = game_obj.token_to_pid[token]
        logging.info(f"Player {player_pid} connected with token {token}")

    # Add the websocket to the game
    try:
        logging.info(f"Adding websocket for Player {player_pid} to game")
        game_obj.add_websocket(token=token, websocket=websocket)
        logging.info(f"Websocket added for Player {player_pid}")
    except Exception as e:
        logging.error(f"Error adding websocket for Player {player_pid}: {e}")
        logging.error(traceback.format_exc())
        await websocket.close(code=1011)  # Internal error
        return

    try:
        # Keep the connection alive
        while connection_active and not (game_obj.done and game_obj.game_over_messages_sent.get(player_pid, False)):
            try:
                logging.debug(f"Waiting for message from Player {player_pid}")
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)  # 30-second timeout
                logging.info(f"Received data from Player {player_pid}: {data[:100]}...")
                
                try:
                    payload = json.loads(data)
                    command = payload.get("command")
                    logging.info(f"Player {player_pid} sent command: {command}")

                    if command == "leave":
                        logging.info(f"Player {player_pid} requested to leave")
                        await game_obj.set_timeout(token)
                        connection_active = False
                    elif command == "action":
                        logging.info(f"Player {player_pid} sent action: {payload.get('action', '')[:50]}...")
                        
                        # First send acknowledgment before processing the action
                        try:
                            ack_msg = json.dumps({
                                "command": "action_ack",
                                "message": "Action received"
                            })
                            logging.debug(f"Sending action acknowledgment to Player {player_pid}")
                            await websocket.send_text(ack_msg)
                            logging.info(f"Action acknowledgment sent to Player {player_pid}")
                        except Exception as e:
                            logging.error(f"Error sending acknowledgment to Player {player_pid}: {e}")
                            logging.error(traceback.format_exc())
                            raise
                        
                        # Then process the action
                        logging.debug(f"Processing action from Player {player_pid}")
                        await command_action(payload, token, websocket)
                        logging.info(f"Action processed for Player {player_pid}")
                        
                    elif command == "ping":
                        # Respond to ping with pong
                        logging.debug(f"Received ping from Player {player_pid}")
                        await websocket.send_text(json.dumps({"command": "pong"}))
                        logging.debug(f"Sent pong to Player {player_pid}")
                    else:
                        logging.warning(f"Unknown command from Player {player_pid}: {command}")
                        await websocket.send_text(json.dumps({"command": "error", "message": f"Unknown command: {command}"}))
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON from Player {player_pid}: {e}")
                    logging.error(f"Raw data: {data}")
                    await websocket.send_text(json.dumps({"command": "error", "message": "Invalid JSON payload"}))
                except Exception as e:
                    logging.error(f"Error processing command from Player {player_pid}: {e}")
                    logging.error(traceback.format_exc())
                    await websocket.send_text(json.dumps({"command": "error", "message": f"Command processing error: {str(e)}"}))
                    
            except asyncio.TimeoutError:
                # Send a ping to keep the connection alive
                logging.debug(f"No message received from Player {player_pid} in 30s, sending ping")
                try:
                    await websocket.send_text(json.dumps({"command": "ping"}))
                    logging.debug(f"Ping sent to Player {player_pid}")
                except Exception as e:
                    logging.error(f"Error sending ping to Player {player_pid}: {e}")
                    logging.error(traceback.format_exc())
                    connection_active = False
            
            # Check if the game is done and this player has received their game over message
            if game_obj.done and game_obj.game_over_messages_sent.get(player_pid, False):
                logging.info(f"Game is done and Player {player_pid} has received game over message. Breaking loop.")
                break
                
    except WebSocketDisconnect as e:
        logging.warning(f"WebSocket disconnected for Player {player_pid}, code: {e.code}")
        # Don't set timeout immediately if game is already done and we're just waiting for messages
        if not game_obj.done:
            await game_obj.set_timeout(token)
    except Exception as e:
        logging.error(f"Unexpected error for Player {player_pid}: {e}")
        logging.error(traceback.format_exc())
        try:
            await websocket.send_text(json.dumps({"command": "error", "message": f"Server error: {str(e)}"}))
        except:
            logging.error("Could not send error message to client")
    
    # Final cleanup
    logging.info(f"WebSocket handler ending for Player {player_pid}")
    
    # Only set timeout if the game isn't already done
    if not game_obj.done:
        try:
            logging.info(f"Setting timeout for Player {player_pid}")
            await game_obj.set_timeout(token)
            logging.info(f"Timeout set for Player {player_pid}")
        except Exception as e:
            logging.error(f"Error during timeout handling: {e}")
            logging.error(traceback.format_exc())


async def command_action(payload: Dict, token: str, websocket: WebSocket):
    global game_obj
    action = payload.get("action")
    player_pid = game_obj.token_to_pid.get(token, "unknown")
    
    logging.info(f"Processing action command for Player {player_pid}")
    
    if action is None:
        logging.error(f"No action provided by Player {player_pid}")
        await websocket.send_text(json.dumps({"command": "error", "message": "No action provided"}))
        return
        
    if not game_obj.check_turn(token=token):
        logging.error(f"Not Player {player_pid}'s turn to act")
        await websocket.send_text(json.dumps({"command": "error", "message": "Not your turn"}))
        return
    
    # First send acknowledgment before processing the action
    try:
        ack_msg = json.dumps({
            "command": "action_ack",
            "message": "Action received"
        })
        logging.info(f"Sending action acknowledgment to Player {player_pid}")
        await websocket.send_text(ack_msg)
        logging.info(f"Action acknowledgment sent to Player {player_pid}")
    except Exception as e:
        logging.error(f"Error sending acknowledgment: {e}")
    
    # Process the action without closing the connection
    try:
        game_obj.step_env(token=token, action=action)
        logging.info(f"Action processed successfully for Player {player_pid}")
    except Exception as e:
        logging.error(f"Error processing action: {e}")
        # Don't close the connection on error
        try:
            await websocket.send_text(json.dumps({"command": "error", "message": f"Error processing action: {str(e)}"}))
        except:
            pass


if __name__ == "__main__":
    import signal
    
    def signal_handler(sig, frame):
        global server_shutdown_requested
        logging.info(f"Received signal {sig}, initiating shutdown")
        server_shutdown_requested = True
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server with graceful shutdown
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, ws_ping_interval=30, ws_ping_timeout=90)