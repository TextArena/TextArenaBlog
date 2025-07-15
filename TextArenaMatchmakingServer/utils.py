import os, uuid, asyncio, json, time, math, requests, aiohttp
from tabulate import tabulate
from datetime import datetime, timezone
from typing import Dict, List, Optional
from queue import Queue
import numpy as np
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from starlette.websockets import WebSocketState
from supabase import create_client, Client
import random

import textarena as ta

from config import (
    # database 
    SUPABASE_URL,
    SUPABASE_KEY,

    # matchmaking
    HUMANITY_MODEL_ID,
    # STANDARD_MODEL_IDS,
    DEFAULT_TRUESKILL,
    QUEUE_TIME_INTERVALS,
    BACKGROUND_LOOP_INTERVAL,
    DOUBLE_STANDARD_PENALTY,
    NUM_MATCHMAKING_ENVS_PER_LOOP,
    SINGLE_MATCHMAKING_PROB,
    NUM_MATCHMAKING_ATTEMPTS_MP,

    # AWS
    SERVER_POOL_SIZE,
    SERVER_LOOP_INTERVAL,
    CLUSTER_NAME,
    TASK_DEFINITION,
    SUBNET_IDS,
    SECURITY_GROUP_IDS,
    CONTAINER_NAME
)


logger = logging.getLogger()

# boto
import boto3
ecs_client = boto3.client('ecs', region_name="ap-southeast-1")
ec2_client = boto3.client('ec2', region_name="ap-southeast-1")
elbv2_client = boto3.client('elbv2', region_name="ap-southeast-1")

# global dicts
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
active_connections = {} # token: {name....ws}
general_information = {"past_queue_times": []}

# matchmaking registry
environments = supabase.table("environments").select("id, num_players, env_name, active").eq("active", True).execute()
matchmaking_registry = {}
for env_dict in environments.data:
    matchmaking_registry[env_dict["id"]] = {"num_players": env_dict["num_players"], "env_id": env_dict["env_name"], "queue": {}}


# global serverless server vars
matched_players_queue = Queue()
available_server_pool_queue = Queue()
booted_server_pool = []

# create helpers for matchmaking
environment_ids_list = list(matchmaking_registry.keys())
current_env_matchmaking_idx = 0


pending_tasks = {}

# implement ALB registration function in utils.py
def register_fargate_task_to_alb(game_id: str, ip: str, port=8000):
    """
    Register a Fargate task with the Application Load Balancer
    
    Args:
        game_id: Unique identifier for the game (used in URL path)
        ip: Private IP of the Fargate task
        port: Port the game server is running on
        
    Returns:
        Dictionary containing registration details or None if failed
    """
    try:
        # Create a unique target group name with the first 8 chars of game_id
        short_id = game_id[:8]
        unique_suffix = str(uuid.uuid4())[:8]
        target_group_name = f"game-{short_id}-{unique_suffix}"
        
        # Truncate if needed (32 char limit for target group names)
        if len(target_group_name) > 32:
            target_group_name = target_group_name[:32]
        
        # 1. Create target group - REMOVE TargetGroupAttributes
        from config import VPC_ID
        tg_response = elbv2_client.create_target_group(
            Name=target_group_name,
            Protocol='HTTP', 
            Port=port,
            VpcId=VPC_ID,
            TargetType='ip',
            HealthCheckProtocol='HTTP',
            HealthCheckPath='/health',  # Make sure your game servers have this endpoint
            HealthCheckIntervalSeconds=15,
            HealthCheckTimeoutSeconds=5,
            HealthyThresholdCount=2,
            UnhealthyThresholdCount=3
        )
        
        target_group_arn = tg_response['TargetGroups'][0]['TargetGroupArn']
        
        # 1b. Set the target group attributes separately
        elbv2_client.modify_target_group_attributes(
            TargetGroupArn=target_group_arn,
            Attributes=[
                {
                    'Key': 'stickiness.enabled',
                    'Value': 'true'
                },
                {
                    'Key': 'stickiness.type',
                    'Value': 'lb_cookie'
                },
                {
                    'Key': 'stickiness.lb_cookie.duration_seconds',
                    'Value': '3600'  # 1 hour session stickiness
                }
            ]
        )
        
        # 2. Register the IP as a target
        elbv2_client.register_targets(
            TargetGroupArn=target_group_arn,
            Targets=[{"Id": ip, "Port": port}]
        )
        
        # 3. Get existing rules to find available priority
        from config import LISTENER_ARN
        rules_response = elbv2_client.describe_rules(ListenerArn=LISTENER_ARN)
        existing_priorities = [int(rule['Priority']) for rule in rules_response['Rules'] 
                              if rule['Priority'] != 'default' and rule['Priority'].isdigit()]
        
        # Find next available priority
        new_priority = 1
        while new_priority in existing_priorities and new_priority < 50000:
            new_priority += 1
        
        # 4. Create the routing rule for this game
        path_pattern = f"/game/{game_id}*"  # Wildcard to catch subpaths
        rule_response = elbv2_client.create_rule(
            ListenerArn=LISTENER_ARN,
            Conditions=[
                {
                    "Field": "path-pattern",
                    "Values": [path_pattern]
                }
            ],
            Priority=new_priority,
            Actions=[
                {
                    "Type": "forward",
                    "TargetGroupArn": target_group_arn
                }
            ]
        )
        
        rule_arn = rule_response['Rules'][0]['RuleArn']
        
        # 5. Add this mapping to a database for cleanup later
        from config import GAME_DOMAIN
        game_url = f"wss://{GAME_DOMAIN}/game/{game_id}"
        
        # Store ALB routing info in Supabase for later cleanup
        from utils import supabase
        supabase.table("game_routing").insert({
            "game_id": game_id,
            "target_group_arn": target_group_arn,
            "rule_arn": rule_arn,
            "server_ip": ip,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        
        logging.info(f"[ALB] Routing configured: {game_url} → {ip}:{port}")
        
        return {
            "game_id": game_id,
            "target_group_arn": target_group_arn,
            "rule_arn": rule_arn,
            "game_url": game_url
        }
        
    except Exception as e:
        logging.error(f"[ALB] Error setting up routing: {str(e)}")
        return None

# Add clean up function 
async def cleanup_game_routing(game_id):
    """Remove ALB routing resources when a game ends"""
    try:
        # Get routing info from database
        result = supabase.table("game_routing").select("*").eq("game_id", game_id).execute()
        
        if not result.data:
            logging.warning(f"No routing information found for game {game_id}")
            return False
            
        routing_info = result.data[0]
        rule_arn = routing_info.get("rule_arn")
        target_group_arn = routing_info.get("target_group_arn")
        
        # Delete the rule first
        if rule_arn:
            elbv2_client.delete_rule(RuleArn=rule_arn)
            logging.info(f"Deleted ALB rule for game {game_id}")
            
        # Wait briefly to ensure the rule is deleted
        await asyncio.sleep(2)
        
        # Then delete the target group
        if target_group_arn:
            elbv2_client.delete_target_group(TargetGroupArn=target_group_arn)
            logging.info(f"Deleted target group for game {game_id}")
            
        # Update database status
        supabase.table("game_routing").update({
            "status": "completed",
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }).eq("game_id", game_id).execute()
        
        return True
        
    except Exception as e:
        logging.error(f"Error cleaning up ALB resources for game {game_id}: {str(e)}")
        return False

async def periodic_game_server_health_check():
    """
    Periodically check all active game servers and clean up resources for unhealthy ones.
    This runs as a background task on the matchmaking server.
    """
    while True:
        try:
            # Get all active game routings from database
            result = supabase.table("game_routing").select("*").is_("deleted_at", "null").execute()
                
            if not result.data:
                # No active games to check
                await asyncio.sleep(60)  # Check every minute
                continue
                
            logging.info(f"Checking health of {len(result.data)} active game servers")
            
            # Get list of active tasks from ECS
            active_task_arns = []
            paginator = ecs_client.get_paginator('list_tasks')
            for page in paginator.paginate(cluster=CLUSTER_NAME, desiredStatus='RUNNING'):
                active_task_arns.extend(page['taskArns'])
                
            # Get task details to map ARNs to IPs
            active_task_ips = set()
            if active_task_arns:
                # Process tasks in batches of 100 (AWS API limit)
                for i in range(0, len(active_task_arns), 100):
                    batch = active_task_arns[i:i+100]
                    tasks = ecs_client.describe_tasks(cluster=CLUSTER_NAME, tasks=batch)['tasks']
                    
                    for task in tasks:
                        task_ip = get_task_ip(task)
                        if task_ip:
                            active_task_ips.add(task_ip)
            
            # Check each game server
            for game_entry in result.data:
                game_id = game_entry['game_id']
                server_ip = game_entry['server_ip']
                
                # Check if server IP is in active tasks list
                if server_ip not in active_task_ips:
                    logging.info(f"Game server {game_id} (IP: {server_ip}) is no longer active, cleaning up")
                    await cleanup_game_routing(game_id)
                    continue
                    
                # Also try a direct health check
                try:
                    # Set a short timeout since we expect quick response from healthy servers
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                        url = f"http://{server_ip}:8000/health"
                        async with session.get(url) as response:
                            if response.status != 200:
                                logging.info(f"Game server {game_id} (IP: {server_ip}) failed health check, cleaning up")
                                await cleanup_game_routing(game_id)
                except Exception:
                    # Connection error, server is not responding
                    logging.info(f"Game server {game_id} (IP: {server_ip}) is not responding, cleaning up")
                    await cleanup_game_routing(game_id)
                    
        except Exception as e:
            logging.error(f"Error in game server health check: {str(e)}")
            
        # Run every minute
        await asyncio.sleep(60)

# Improved logging configuration
def setup_logging():
    log_format = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation
            RotatingFileHandler(
                "textarena_matchmaking.log", 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Create logger
    logger = logging.getLogger("textarena")
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

# Add this to utils.py, where the matched_players_queue is defined or used

def log_matchmaking_queue_state():
    """Log the current state of the matchmaking queue for debugging"""
    if matched_players_queue.empty():
        logging.info("[QUEUE_STATUS] matched_players_queue is empty")
    else:
        # Since Queue doesn't support inspection without removing items, we'll log only the size
        queue_size = matched_players_queue.qsize()
        logging.info(f"[QUEUE_STATUS] matched_players_queue has {queue_size} pending matches")


# join / leave queue 
def join_matchmaking(token: str, env_ids: List[int], model_id: int, human_id: int):
    """ Add a given connection token into the matchmaking queue for each environment ID in env_ids """
    global matchmaking_registry

    is_human = (model_id == HUMANITY_MODEL_ID)
    is_standard = (model_id in [item["id"] for item in supabase.table("models").select("id").eq("is_standard", True).order("id").execute().data])

    # Fetch or default to default TrueSkill
    if model_id == 0:
        # Human player
        trueskill_response = supabase.rpc("get_latest_trueskill", {
            "p_model_id": 0,
            "p_human_id": human_id,
            "p_env_ids": env_ids
        }).execute()
    else:
        # Model player
        trueskill_response = supabase.rpc("get_latest_trueskill", {
            "p_model_id": model_id,
            "p_human_id": 0,  # human_id ignored when model_id != 0
            "p_env_ids": env_ids
        }).execute()

    trueskill_map = {r["out_environment_id"]: r["out_trueskill"] for r in (trueskill_response.data or [])}

    for env_id in env_ids:
        trueskill = trueskill_map.get(env_id, DEFAULT_TRUESKILL)
        if env_id in matchmaking_registry:
            matchmaking_registry[env_id]["queue"][token] = {
                "token": token,
                "model_id": model_id,
                "trueskill": trueskill,
                "is_human": is_human,
                "is_standard": is_standard,
                "timestamp": time.time()
            }


def leave_matchmaking(token: str):
    """ Remove a token from all environment queues """
    global matchmaking_registry
    for env_id in matchmaking_registry:
        if token in matchmaking_registry[env_id]["queue"]:
            del matchmaking_registry[env_id]["queue"][token]


def calculate_match_probability(candidates):
    model_a, model_b = candidates

    # skip if both are humans (commented out)
    if model_a["is_human"] and model_b["is_human"]:
        return 0

    current_time = time.time()

    # apply the double penalty only if both are standard and non-human
    standard_nonhuman_a = model_a["is_standard"] and not model_a["is_human"]
    standard_nonhuman_b = model_b["is_standard"] and not model_b["is_human"]

    # "time_component" can increase the longer they wait
    if standard_nonhuman_a and standard_nonhuman_b:
        time_component = DOUBLE_STANDARD_PENALTY
        logging.info(f"Time Component: Both standard non-human models, using DOUBLE_STANDARD_PENALTY = {time_component}")
    else:
        queue_times = [current_time - m["timestamp"] for m in (model_a, model_b) if not m["is_standard"]]
        highest_queue_time = max(queue_times) if queue_times else 0
        time_component = 1 + highest_queue_time / QUEUE_TIME_INTERVALS * 0.1
        logging.info(f"Time Component: highest_queue_time = {highest_queue_time:.3f} sec, "
                     f"QUEUE_TIME_INTERVALS = {QUEUE_TIME_INTERVALS}, "
                     f"time_component = 1 + {highest_queue_time:.3f} / {QUEUE_TIME_INTERVALS} * 0.1 = {time_component:.3f}")

    # "trueskill_component" exponential diff in trueskills scores
    trueskill_diff = abs(model_a["trueskill"] - model_b["trueskill"]) / 20
    trueskill_component = (1 - trueskill_diff) ** 3
    logging.info(f"TrueSkill Component: model_a_trueskill = {model_a['trueskill']}, "
                 f"model_b_trueskill = {model_b['trueskill']}, "
                 f"diff = {trueskill_diff:.3f}, "
                 f"trueskill_component = (1 - {trueskill_diff:.3f})^3 = {trueskill_component:.3f}")

    score = time_component * trueskill_component
    logging.info(f"MATCHMAKING SCORE: {model_a['model_id']} vs {model_b['model_id']} => {score:.3f} "
                 f"(time_component = {time_component:.3f} * trueskill_component = {trueskill_component:.3f})")
    return max(score, 0.00001)  # Ensure the score is never zero


def calculate_multiplayer_match_probability(candidates, num_players_required):
    current_time = time.time()

    avg_trueskill = np.mean([player['trueskill'] for player in candidates])
    trueskill_delta = np.mean([abs(avg_trueskill - player['trueskill']) for player in candidates]) / 20
    trueskill_component = (1 - trueskill_delta) ** 3

    # only penalize if all are standard and non-human
    all_standard_nonhuman = all(p["is_standard"] and not p["is_human"] for p in candidates)

    if not all_standard_nonhuman:
        timestamps_to_check = [player["timestamp"] for player in candidates if not player["is_standard"] or player["is_human"]]
        earliest_timestamp = min(timestamps_to_check)
        time_component = 1 + (current_time - earliest_timestamp) / QUEUE_TIME_INTERVALS * 0.1
        logging.info(f"Time Component: earliest_timestamp = {current_time - earliest_timestamp:.3f} sec, "
                     f"QUEUE_TIME_INTERVALS = {QUEUE_TIME_INTERVALS}, "
                     f"time_component = 1 + Δt / {QUEUE_TIME_INTERVALS} * 0.1 = {time_component:.3f}")
    else:
        time_component = DOUBLE_STANDARD_PENALTY
        logging.info(f"Time Component: All standard non-human models, using DOUBLE_STANDARD_PENALTY = {time_component}")

    num_players_component = 1 + (num_players_required / 15)

    score = time_component * trueskill_component * num_players_component
    logging.info(f"MATCHMAKING SCORE: {', '.join([str(model['model_id']) for model in candidates])} => {score:.3f} "
                 f"(time_component = {time_component:.3f} * trueskill_component = {trueskill_component:.3f} * num_players_component = {num_players_component:.3f})")

    return max(score, 0.00001)  # Ensure the score is never zero


async def run_matchmaking():
    global matchmaking_registry, general_information, environment_ids_list
    global matched_players_queue, current_env_matchmaking_idx

    matched_user_tokens = set()

    for _ in range(NUM_MATCHMAKING_ENVS_PER_LOOP):
        environment_id = environment_ids_list[current_env_matchmaking_idx]
        current_env_matchmaking_idx = (current_env_matchmaking_idx+1)%len(environment_ids_list)

        env_registry = matchmaking_registry[environment_id]

        num_players_required = env_registry["num_players"]
        queue_values = list(env_registry["queue"].values())
        # print(queue_values)
        env_id = env_registry["env_id"]

        logging.info(f"Trying to match make {num_players_required} player(s) for env {env_id}.")

        # check for any single player games in the env_ids
        if num_players_required == 1:

            # 1 player game
            if len(queue_values) < 1:
                continue

            for i in range(len(queue_values) - 1):
                a = queue_values[i]
                if a["token"] in matched_user_tokens:
                    continue

                match_prob = SINGLE_MATCHMAKING_PROB

                if a['is_standard']:
                    match_prob *= DOUBLE_STANDARD_PENALTY
                    
                if np.random.uniform() < match_prob:
                    # Yay! We found a match
                    if not a["is_standard"]:
                        general_information["past_queue_times"].extend([
                            time.time() - a["timestamp"]
                        ])

                    # Add to match instantiation queue
                    matched_players_queue.put({
                        "environment_id": environment_id,
                        "env_id": env_id,
                        "player_tokens": (a["token"])
                    })

                    # log the match creation
                    logging.info(f"[MATCH_CREATED] Players {a['token']} matched for env {env_id}")
                    log_matchmaking_queue_state()

                    # Add to matched user tokens
                    matched_user_tokens.add(a["token"])
                    
                    break  # Exit this loop, as player 'a' is now matched
                    
        elif num_players_required == 2:
            # 2 player game 
            if len(queue_values) < 2:
                continue 

            # sort by trueskill
            queue_values.sort(key=lambda user: user["trueskill"])

            for i in range(len(queue_values) - 1):
                a = queue_values[i]
                if a["token"] in matched_user_tokens:
                    continue
                
                for j in range(i+1, min(i+10, len(queue_values))):
                    b = queue_values[j]
                    if b["token"] in matched_user_tokens:
                        continue
                    
                    # Calculate match probability
                    match_prob = calculate_match_probability((a, b))
                    if match_prob > 0:
                        # Try to get game
                        if np.random.uniform() < match_prob:
                            # Yay! We found a match
                            general_information["past_queue_times"].extend([
                                time.time() - p["timestamp"]
                                for p in (a, b)
                                if not p["is_standard"] or p["is_human"]
                            ])

                            # Add to match instantiation queue
                            matched_players_queue.put({
                                "environment_id": environment_id,
                                "env_id": env_id,
                                "player_tokens": (a["token"], b["token"])
                            })

                            # log the match creation
                            logging.info(f"[MATCH_CREATED] Players {a['token']} and {b['token']} matched for env {env_id}")
                            log_matchmaking_queue_state()

                            # Add to matched user tokens
                            matched_user_tokens.add(a["token"])
                            matched_user_tokens.add(b["token"])
                            
                            # Break out of the inner loop once player a is matched
                            break  # Exit this loop, as player 'a' is now matched
                    
                # Also check again after inner loop to see if player 'a' got matched
                if a["token"] in matched_user_tokens:
                    continue

        # Fix for the multiplayer section
        else:  # num_players_required > 2
            
            # If not enough players, skip this environment
            if len(queue_values) < num_players_required:
                logging.info(f"[MULTIPLAYER] Not enough players in queue ({len(queue_values)}/{num_players_required})")
                continue

            # Sort by trueskill for better matching
            queue_values.sort(key=lambda user: user['trueskill'])

            # Iterate through potential anchor players
            for i in range(len(queue_values) - (num_players_required - 1)):
                anchor_player = queue_values[i]
                
                # Skip if anchor player already matched
                if anchor_player["token"] in matched_user_tokens:
                    logging.info(f"[MULTIPLAYER] Anchor player {anchor_player['token']} already matched, skipping")
                    continue
                
                # Create a pool of potential match candidates
                # Include all available players that aren't already matched
                sample_pool = []
                for k in range(i + 1, len(queue_values)):
                    if queue_values[k]["token"] not in matched_user_tokens:
                        sample_pool.append(queue_values[k])
                        
                        # Once we have enough candidates, we can stop adding more
                        if len(sample_pool) >= 5 + num_players_required:
                            break
                
                logging.info(f"[MULTIPLAYER] Sample pool size: {len(sample_pool)}")
                
                # We need at least (num_players_required - 1) more players to form a match
                if len(sample_pool) < num_players_required - 1:
                    logging.info(f"[MULTIPLAYER] Not enough remaining players in sample pool")
                    continue

                # Try multiple combinations to find a good match
                for j in range(NUM_MATCHMAKING_ATTEMPTS_MP):
                    # Select (num_players_required - 1) players from sample pool
                    other_players = random.sample(sample_pool, num_players_required - 1)
                    all_players = [anchor_player] + other_players  # Combine anchor player with others
                    
                    # Calculate match probability
                    match_prob = calculate_multiplayer_match_probability(all_players, num_players_required)
                    logging.info(f"[MULTIPLAYER] Match probability: {match_prob}")

                    # Try to match based on probability
                    if np.random.uniform() < match_prob:
                        # Success! We found a match
                        
                        # Update queue time stats for non-standard players
                        general_information["past_queue_times"].extend([
                            time.time() - p["timestamp"]
                            for p in all_players
                            if not p["is_standard"] or p["is_human"]
                        ])

                        # Collect all player tokens for the match
                        player_tokens = [player["token"] for player in all_players]
                        
                        # Add to match instantiation queue
                        matched_players_queue.put({
                            "environment_id": environment_id,
                            "env_id": env_id,
                            "player_tokens": player_tokens
                        })

                        # Log the match creation
                        logging.info(f"[MATCH_CREATED] Players {player_tokens} matched for env {env_id}")
                        log_matchmaking_queue_state()

                        # Add all matched players to matched_user_tokens
                        for token in player_tokens:
                            matched_user_tokens.add(token)
                        
                        # Break out of the attempts loop since we found a match
                        break

        # remove from matchmaking_registry
        for token in matched_user_tokens:
            leave_matchmaking(token)


# matchmaking loop
async def matchmaking_loop():
    while True:
        try:
            t0 = time.time()
            await run_matchmaking()
            elapsed = time.time() - t0
            logging.info(f"[matchmaking_loop] took {elapsed:.2f}s")
        except Exception as exc:
            logging.error(f"[matchmaking_loop] error: {exc}")

        await asyncio.sleep(BACKGROUND_LOOP_INTERVAL)


def get_task_ip(task):
    try:
        for attachment in task.get('attachments', []):
            if attachment['type'] == 'ElasticNetworkInterface':
                eni_id = None
                for detail in attachment['details']:
                    if detail['name'] == 'networkInterfaceId':
                        eni_id = detail['value']
                        break

                if eni_id:
                    eni_response = ec2_client.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
                    if eni_response['NetworkInterfaces']:
                        network_interface = eni_response['NetworkInterfaces'][0]
                        # Return the private IP instead of public IP
                        private_ip = network_interface.get('PrivateIpAddress')
                        if private_ip:
                            return private_ip
                        else:
                            print(f"[WARN] Private IP not found for ENI.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to get private IP: {e}")
        return None


def spin_up_new_instance() -> bool:
    global booted_server_pool
    try:
        response = ecs_client.run_task(
            cluster=CLUSTER_NAME,
            taskDefinition=TASK_DEFINITION,
            count=1,
            launchType="FARGATE",
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": SUBNET_IDS,
                    "securityGroups": SECURITY_GROUP_IDS,
                    "assignPublicIp": "ENABLED"
                }
            },
            startedBy=f"textarena_server_{int(time.time())}",
            overrides={"containerOverrides": [{"name": CONTAINER_NAME}]}
        )

        if 'tasks' not in response or not response['tasks']:
            logging.warning("No tasks returned by ECS")
            return False

        task_arn = response['tasks'][0]['taskArn']
        task_id = task_arn.split('/')[-1]

        # Wait until the task is ready and has a public IP
        for _ in range(10):  # max ~10s wait
            time.sleep(1)
            task = ecs_client.describe_tasks(cluster=CLUSTER_NAME, tasks=[task_id])['tasks'][0]
            public_ip = get_task_ip(task)
            if public_ip:
                booted_server_pool.append(public_ip)
                logging.info(f"[spin_up_new_instance] Booted server: {public_ip}")
                return True

        logging.warning(f"[spin_up_new_instance] Timeout: No public IP for task {task_id}")
        return False

    except Exception as e:
        logging.error(f"[spin_up_new_instance] Exception: {e}")
        return False


async def handle_server_setup_and_connection(server_ip, match_information):
    global active_connections

    # Generate a unique game ID
    game_id = str(uuid.uuid4())
    
    # Register with ALB first
    alb_registration = register_fargate_task_to_alb(game_id, server_ip)
    
    if not alb_registration:
        logging.error(f"Failed to register game server with ALB: {server_ip}")
        return
        
    # Store game mapping
    game_url = alb_registration["game_url"]
    
    # Initialize server
    url = f"http://{server_ip}:8000/initialize"
    payload = {
        "environment_id": match_information["environment_id"],
        "env_id": match_information["env_id"], 
        "tokens": list(match_information["player_tokens"]),
        "game_id": game_id  # Pass the game_id to the server
    }

    # Send initialization to the game server
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        logging.error(f"Error initializing server: {response.status_code} {response.text}")
        return

    # Create client payload with game URL instead of server IP
    client_payload = {
        "command": "match_found", 
        "game_id": game_id,
        "game_url": game_url.replace("wss://", ""),  # just the domain part
        "env_id": match_information["env_id"], 
        "environment_id": match_information["environment_id"]
    }

    # Send match info to all matched players
    for token in match_information["player_tokens"]:
        if token in active_connections:
            websocket = active_connections[token]["ws"]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(client_payload))
                await websocket.close()
            del active_connections[token]


async def run_server_allocation():
    global available_server_pool_queue, booted_server_pool

    # Step 1: Check booted servers for health and move healthy ones to available pool
    available_server_ips = []
    async with aiohttp.ClientSession() as session:
        for server_ip in booted_server_pool:
            url = f"http://{server_ip}:8000/health"
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("status") == "ok":
                            available_server_ips.append(server_ip)
            except Exception as e:
                print(f"Server {server_ip} is not responding. Error: {e}")

    # Move healthy servers to the available queue
    for server_ip in available_server_ips:
        booted_server_pool.remove(server_ip)
        available_server_pool_queue.put(server_ip)

    # Step 2: Calculate total servers (healthy + still booting)
    total_known_servers = available_server_pool_queue.qsize() + len(booted_server_pool)

    # Step 3: Spin up new servers only if needed
    while total_known_servers < SERVER_POOL_SIZE:
        spin_up_new_instance()
        total_known_servers += 1

    # Step 4: Allocate servers to matched players
    while not matched_players_queue.empty():
        if available_server_pool_queue.qsize() > 0:
            server_ip = available_server_pool_queue.get()
            match_information = matched_players_queue.get()
            await handle_server_setup_and_connection(server_ip, match_information)
        else:
            break


async def replenish_server_pool():
    while True:
        try:
            t0 = time.time()
            await run_server_allocation()
            elapsed = time.time() - t0
            logging.info(f"[replenish_server_pool] took {elapsed:.2f}s")
        except Exception as exc:
            logging.error(f"[replenish_server_pool] error: {exc}")

        await asyncio.sleep(SERVER_LOOP_INTERVAL)