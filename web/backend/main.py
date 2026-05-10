import asyncio
import os
import json
import subprocess
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import dotenv_values, set_key

app = FastAPI(title="XAUUSD Trading Bot Controller")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")

# Process state
current_process: Optional[asyncio.subprocess.Process] = None
process_logs: List[str] = []
active_connections: List[WebSocket] = []

class ConfigUpdate(BaseModel):
    env_vars: Optional[Dict[str, str]] = None
    model_data: Optional[Dict[str, Any]] = None

class RunRequest(BaseModel):
    script: str
    args: Optional[List[str]] = []

@app.get("/api/config")
def get_config():
    # Read .env
    env_data = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    
    # Read model_config.json
    model_data = {}
    if os.path.exists(MODEL_CONFIG_PATH):
        try:
            with open(MODEL_CONFIG_PATH, "r") as f:
                model_data = json.load(f)
        except Exception as e:
            model_data = {"error": str(e)}
            
    return {"env": env_data, "model": model_data}

@app.post("/api/config")
def update_config(update: ConfigUpdate):
    if update.env_vars:
        if not os.path.exists(ENV_PATH):
            open(ENV_PATH, 'a').close()
        for key, value in update.env_vars.items():
            set_key(ENV_PATH, key, str(value))
            
    if update.model_data:
        if os.path.exists(MODEL_CONFIG_PATH):
            with open(MODEL_CONFIG_PATH, "r") as f:
                current_model_data = json.load(f)
            current_model_data.update(update.model_data)
            with open(MODEL_CONFIG_PATH, "w") as f:
                json.dump(current_model_data, f, indent=4)
        else:
            with open(MODEL_CONFIG_PATH, "w") as f:
                json.dump(update.model_data, f, indent=4)
                
    return {"status": "success"}

async def broadcast_log(message: str):
    process_logs.append(message)
    # Keep last 1000 lines
    if len(process_logs) > 1000:
        process_logs.pop(0)
    
    dead_connections = []
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            dead_connections.append(connection)
            
    for connection in dead_connections:
        active_connections.remove(connection)

async def run_process(script_path: str, args: List[str]):
    global current_process
    cmd = ["python", script_path] + args
    cmd_str = " ".join(cmd)
    
    await broadcast_log(f"--- Starting Process: {cmd_str} ---")
    
    try:
        current_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_ROOT
        )
        
        while current_process and current_process.stdout:
            line = await current_process.stdout.readline()
            if not line:
                break
            await broadcast_log(line.decode().rstrip())
            
        await current_process.wait()
        await broadcast_log(f"--- Process finished with exit code {current_process.returncode} ---")
    except Exception as e:
        await broadcast_log(f"--- Process error: {str(e)} ---")
    finally:
        current_process = None

@app.post("/api/run")
async def start_script(req: RunRequest):
    global current_process
    
    if current_process:
        raise HTTPException(status_code=400, detail="A process is already running. Please stop it first.")
        
    script_path = os.path.join(PROJECT_ROOT, req.script)
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"Script not found: {req.script}")
        
    # Start process in background task
    asyncio.create_task(run_process(req.script, req.args or []))
    
    return {"status": "started", "script": req.script}

@app.post("/api/stop")
async def stop_script():
    global current_process
    if current_process:
        try:
            current_process.terminate()
            await broadcast_log("--- Process termination requested ---")
            return {"status": "stopping"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "not_running"}

@app.get("/api/status")
def get_status():
    return {
        "is_running": current_process is not None
    }

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    # Send historical logs on connect
    for log_line in process_logs[-100:]:
        await websocket.send_text(log_line)
        
    try:
        while True:
            # Just keep connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
