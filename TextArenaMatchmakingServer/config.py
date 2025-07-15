import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # Load environment variables from .env

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# App-specific constants
NUM_CONNECTION_LIMIT = 5000

# Elo-related constants
HUMANITY_MODEL_ID = 0

# Matchmaking constants
DEFAULT_TRUESKILL = 25
QUEUE_TIME_INTERVALS = 10
BACKGROUND_LOOP_INTERVAL = 1
DOUBLE_STANDARD_PENALTY = 0.00001

NUM_MATCHMAKING_ENVS_PER_LOOP = 10

# Single player matchmaking constants
SINGLE_MATCHMAKING_PROB = 0.25

# Multi PLayer matchmaking constants
NUM_MATCHMAKING_ATTEMPTS_MP = 10

# Server pool config
SERVER_POOL_SIZE = 3
SERVER_LOOP_INTERVAL = 3

# AWS stuff
CLUSTER_NAME = ...
TASK_DEFINITION = ...
SUBNET_IDS = ["subnet-1234567890abcdefg", "subnet-2345678901abcdefg", "subnet-3456789012abcdefg"]
SECURITY_GROUP_IDS = ["sg-0123456789abcdefg"] 
CONTAINER_NAME = ...

VPC_ID = "vpc-0123456789abcdefg"
ALB_ARN = ...
LISTENER_ARN = ...
GAME_DOMAIN = ...
