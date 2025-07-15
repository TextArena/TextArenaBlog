# api.py
from fastapi import APIRouter, Body, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from typing import Optional
import uuid
import numpy as np

## new imports
from pydantic import BaseModel, EmailStr
import bcrypt

from utils import supabase, general_information, active_connections, matchmaking_registry, logging
from config import NUM_CONNECTION_LIMIT, HUMANITY_MODEL_ID

router = APIRouter()

class RegisterModel(BaseModel):
    email: EmailStr
    password: str
    human_name: str
    cookie_id: str
    agree_to_terms: bool = True  # Default to True to handle existing clients
    agree_to_marketing: bool = False  # Default to False to handle existing clients

class ModelRegistration(BaseModel):
    model_name: str
    description: str
    email: str
    model_token: Optional[str] = None  # Add this field for deterministic tokens

class LoginModel(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
async def register(data: RegisterModel, request: Request, response: Response):
    try:
        # Log the request for debugging
        origin = request.headers.get("origin", "")
        logging.info(f"Register request from origin: {origin}")
        
        # Check if email already exists
        existing_email = supabase.table("humans").select("id").eq("email", data.email).execute()
        if existing_email.data:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Check if human_name already exists
        existing_username = supabase.table("humans").select("id").eq("human_name", data.human_name).execute()
        if existing_username.data:
            raise HTTPException(status_code=400, detail="Username already taken")

        # Hash password
        hashed_pw = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()

        # Check if cookie_id exists
        cookie_check = supabase.table("humans").select("id").eq("cookie_id", data.cookie_id).execute()
        
        try:
            if not cookie_check.data:
                # If not, create a new record instead of updating
                result = supabase.table("humans").insert({
                    "cookie_id": data.cookie_id,
                    "email": data.email,
                    "password": hashed_pw,
                    "human_name": data.human_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "last_active": datetime.now(timezone.utc).isoformat(),
                    "agree_to_terms": data.agree_to_terms,
                    "agree_to_marketing": data.agree_to_marketing
                }).execute()
            else:
                # Update existing record
                result = supabase.table("humans").update({
                    "email": data.email,
                    "password": hashed_pw,
                    "human_name": data.human_name,
                    "last_active": datetime.now(timezone.utc).isoformat(),
                    "agree_to_terms": data.agree_to_terms,
                    "agree_to_marketing": data.agree_to_marketing
                }).eq("cookie_id", data.cookie_id).execute()
                
            # Check for Supabase errors without accessing .error attribute
            if not result.data:
                raise HTTPException(status_code=500, detail="Failed to create/update profile")
        except Exception as internal_err:
            logging.error(f"Supabase error in register: {internal_err}")
            raise HTTPException(status_code=500, detail="Database operation failed")

        # Set cookie in response
        response.set_cookie(
            key="user_id",
            value=data.cookie_id,
            httponly=False,
            max_age=31536000,
            path="/",
            samesite="lax",
            secure=False
        )

        # Return user data along with token for consistency with login
        return {
            "token": data.cookie_id,
            "user": {
                "email": data.email,
                "human_name": data.human_name
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error in register: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login")
async def login(data: LoginModel, request: Request, response: Response):
    try:
        # Log the request for debugging
        origin = request.headers.get("origin", "")
        logging.info(f"Login request from origin: {origin}")
        
        try:
            result = supabase.table("humans").select("email", "password", "cookie_id", "human_name").eq("email", data.email).single().execute()
        except Exception as db_error:
            logging.error(f"Database error in login: {db_error}")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not result.data:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = result.data
        if not bcrypt.checkpw(data.password.encode(), user["password"].encode()):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last active timestamp
        try:
            supabase.table("humans").update({
                "last_active": datetime.now(timezone.utc).isoformat()
            }).eq("cookie_id", user["cookie_id"]).execute()
        except Exception as update_error:
            logging.warning(f"Could not update last_active: {update_error}")
            # Continue anyway since this isn't critical

        # Set cookie in response
        response.set_cookie(
            key="user_id",
            value=user["cookie_id"],
            httponly=False,
            max_age=31536000,
            path="/",
            samesite="lax",
            secure=False
        )

        # Return user data along with token
        return {
            "token": user["cookie_id"],
            "user": {
                "email": user["email"],
                "human_name": user["human_name"]
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error in login: {e}")
        raise HTTPException(status_code=401, detail="Login failed")


@router.post("/logout")
async def logout(response: Response):
    # Clear the cookie
    response.delete_cookie(key="user_id", path="/")
    
    return {"message": "Logged out"}


@router.post("/save_game")
async def save_game(game_data: dict):
    try:
        game_id = game_data["metadata"]["game_id"]

        # Check if game already exists
        existing_game = supabase.table("shared_games").select("game_id").eq("game_id", game_id).execute()

        # If it doesn't exist, insert
        if not existing_game.data:
            supabase.table("shared_games").insert({
                "game_id": game_id,
                "data": game_data,
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute()

        return {"url": f"/game/{game_id}"}
    except Exception as e:
        logging.error(f"Error in save_game: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/games/{game_id}")
async def get_game(game_id: str):
    try:
        result = supabase.table("shared_games").select("data").eq("game_id", game_id).single().execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Game not found")
        return result.data["data"]
    except Exception as e:
        logging.error(f"Error in get_game: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/init")
async def init_endpoint(request: Request, response: Response):
    print('Initializing session')
    
    # Try to get token from different sources
    token = None
    
    # 1. Check cookie first
    cookie_token = request.cookies.get("user_id")
    if cookie_token:
        token = cookie_token
        logging.info(f"Found existing token in cookie: {token}")
    
    # 2. Check header if no cookie
    if not token:
        header_token = request.headers.get("X-User-Token")
        if header_token:
            token = header_token
            logging.info(f"Found existing token in header: {token}")
    
    # 3. Generate new if neither exists
    if not token:
        token = str(uuid.uuid4())
        logging.info(f"Generated new token: {token}")
    
    # Always set cookie (even if we got token from header)
    # With modified settings for HTTP
    response.set_cookie(
        key="user_id",
        value=token,
        httponly=False,
        max_age=31536000,
        path="/",
        samesite="lax",  # Changed from "none" for better HTTP compatibility
        secure=False     # Changed from True since we're using HTTP
    )

    # Check/Insert in DB
    now = datetime.now(timezone.utc).isoformat()
    r = supabase.table("humans").select("cookie_id", "email", "human_name", "id").eq("cookie_id", token).execute()
    
    isAuthenticated = False
    user = None
    
    if not r.data:
        supabase.table("humans").insert({
            "cookie_id": token,
            "created_at": now,
            "last_active": now
        }).execute()
        logging.info("Inserted new human")
    else:
        # Check if this is an authenticated user
        if r.data[0].get("email"):
            isAuthenticated = True
            user = {
                "email": r.data[0].get("email"),
                "human_name": r.data[0].get("human_name")
            }
        
        supabase.table("humans").update({"last_active": now}).eq("cookie_id", token).execute()
        logging.info("Updated last_active for user")

    data = {
        "token": token,
        "isAuthenticated": isAuthenticated,
        "user": user,
        "avg_queue_time": float(np.mean(general_information["past_queue_times"])) if general_information["past_queue_times"] else 69,
        "num_players_in_queue": len(active_connections),
        "allow_connection": len(active_connections) < NUM_CONNECTION_LIMIT
    }
    return data


@router.get("/check_matchmaking")
async def check_matchmaking():
    return {
        "avg_queue_time": float(np.mean(general_information["past_queue_times"])) if general_information["past_queue_times"] else 69,
        "num_players_in_queue": len(active_connections),
        "allow_connection": len(active_connections) < NUM_CONNECTION_LIMIT
    }


@router.post("/register_model")
async def register_model(
    model_name: Optional[str] = Body(...),
    description: Optional[str] = Body(None),
    email: Optional[str] = Body(...),
    model_token: Optional[str] = Body(None)
):
    # Check if model_name is already registered
    existing = supabase.table("models").select("id, description, email, model_token").eq("model_name", model_name).execute()
    if existing.data:
        em = existing.data[0]
        email_match = em.get("email") == email
        token_match = em.get("model_token") == model_token

        if email_match and token_match:
            return JSONResponse({
                "model_name": model_name,
                "model_token": em.get("model_token")
            })

        mismatches = []
        if not email_match:
            mismatches.append(f"email (existing: '{em.get('email')}', provided: '{email}')")
        elif not token_match:
            mismatches.append("model token (different token - agent configuration changed)")
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_name}' already registered with different {', '.join(mismatches)}. Use a different model name or ensure all parameters match exactly."
        )

    # Require model_token for new registration
    if not model_token:
        raise HTTPException(
            status_code=400,
            detail="Model token is required for new model registrations. Please provide an agent object when calling make_online() to generate deterministic tokens."
        )

    # Validate UUID5
    try:
        parsed_uuid = uuid.UUID(model_token)
        if parsed_uuid.version != 5:
            raise HTTPException(status_code=400, detail="Invalid token format. Only UUID5 is accepted.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid token format. Must be a valid UUID5.")

    # Insert into models table
    try:
        supabase.table("models").insert({
            "model_name": model_name,
            "description": description,
            "email": email,
            "model_token": model_token,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        logging.error(f"Database error inserting model: {e}")
        raise HTTPException(status_code=500, detail="Failed to register model in database")

    logging.info(f"✅ Model '{model_name}' registered for identifier '{email}' with token '{model_token}'")

    return JSONResponse({
        "model_name": model_name,
        "model_token": model_token,
        "message": "Model registered successfully"
    })

