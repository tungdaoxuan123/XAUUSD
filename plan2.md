Web Interface Design for XAUUSD Trading Bot

  Objective
  Create a modern web interface that allows the user to easily customize all attributes of the XAUUSD trading bot (MT5 credentials, risk
  parameters, strategy configurations) and execute training or live trading scripts defined in RUN_GUIDE.md.

  Architecture Choice: Modern (FastAPI + React)
   - Backend: FastAPI (Python). Since the existing project is in Python, FastAPI provides an incredibly fast and typed REST API to interface
     with the local file system and execute Python subprocesses.
   - Frontend: React (via Vite) + Tailwind CSS. Provides a highly responsive UI with excellent form state management and real-time WebSocket
     capabilities for streaming logs.

  Scope & Impact
  This implementation will not modify the core trading logic but will act as a wrapper/controller.
   - Backend Location: web/backend/
   - Frontend Location: web/frontend/
   - Integrations: Reads and writes to .env, config/model_config.json, and potentially config.py (though modifying .env is preferred for
     overrides). Executes scripts like train_pipeline/train_ensemble_gpu.py.

  Core Features & UI Sections

  1. Configuration Management (The "Customize Any Attribute" Requirement)
  The UI will have forms to manage:
   - MT5 Credentials (.env): MT5_LOGIN, MT5_PASSWORD, MT5_SERVER.
   - Risk Management (.env & config.py): RISK_PER_TRADE_PCT, MAX_DAILY_LOSS_PCT, MAX_TOTAL_LOSS_PCT, LOT_SIZE.
   - Strategy Attributes (model_config.json): Model types, algorithms, timeframe, scaled profit-taking, and confidence sizing.

  2. Pipeline Execution (RUN_GUIDE.md)
  A dedicated section to trigger the commands from your run guide.
   - Form Inputs for Pipeline Arguments: Input fields for --side, --zscore-window, --lookback, --threshold, etc.
   - Execute Buttons: 
     - "Run Training" -> executes python train_pipeline/train_ensemble_gpu.py ...
     - "Run Walk-Forward Backtest" -> executes python train_pipeline/walk_forward_backtest.py ...
   - Live Trading: Start/Stop button for live_sota_trading.py or live_ensemble_trading.py.

  3. Real-Time Logs & Monitoring
   - A terminal-like component in the UI.
   - The FastAPI backend will use WebSockets to stream the stdout/stderr of running processes and tail the ftmo_trading.log file so you can
     watch what happens without needing a shell.

  Implementation Steps

   1. Backend Setup:
      - Initialize a FastAPI app.
      - Create API endpoints GET /api/config and POST /api/config (uses python-dotenv to update .env and json module for model_config.json).
      - Create API endpoints POST /api/run/{script_name} to spawn subprocess.Popen.
      - Create a WebSocket endpoint ws://logs to yield stdout lines from active processes.

   2. Frontend Setup:
      - Initialize Vite React project.
      - Install Tailwind CSS and component libraries (e.g., shadcn/ui or simple custom Tailwind components).
      - Build layout with sidebar navigation (Settings, Training, Live Trading).

   3. Form Integration:
      - Build the Settings forms. On submit, send a POST to the backend to rewrite local config files.

   4. Process Control Integration:
      - Build the UI to construct the CLI arguments based on form inputs and hit the run endpoint. 

  Verification
   - Change a setting in the UI and manually inspect .env to ensure it saved correctly.
   - Click "Run Training" and verify the WebSocket terminal displays the expected output from the Python script.