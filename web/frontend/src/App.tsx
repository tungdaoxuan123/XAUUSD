import { useState, useEffect, useRef } from 'react'
import { Activity, Settings, Play, SquareTerminal, Save } from 'lucide-react'
import { cn } from './lib/utils'

export default function App() {
  const [activeTab, setActiveTab] = useState('settings')
  const [config, setConfig] = useState({ env: {} as Record<string, string>, model: {} as Record<string, any> })
  const [logs, setLogs] = useState<string[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Pipeline Arguments
  const [trainArgs, setTrainArgs] = useState({
    data: 'train_pipeline/data/events_events_long_labeled.csv',
    out_dir: 'train_pipeline/models_gpu_long_lb15',
    side: 'long',
    lookback: '15',
    zscore_window: '250',
    use_gpu: true,
    recency_weight: true
  })

  const [backtestArgs, setBacktestArgs] = useState({
    data: 'train_pipeline/data/events_events_long_labeled.csv',
    model_dir: 'train_pipeline/models_gpu_long_lb15',
    side: 'long',
    lookback: '15',
    threshold: '0.55'
  })

  useEffect(() => {
    fetch('http://localhost:8000/api/config')
      .then(res => res.json())
      .then(data => setConfig({
        env: data.env || {},
        model: data.model || {}
      }))
      .catch(err => console.error("Failed to load config:", err))

    fetch('http://localhost:8000/api/status')
      .then(res => res.json())
      .then(data => setIsRunning(data.is_running))
      .catch(err => console.error("Failed to load status:", err))
      
    const ws = new WebSocket('ws://localhost:8000/ws/logs')
    ws.onmessage = (event) => {
      setLogs(prev => [...prev, event.data])
    }
    return () => ws.close()
  }, [])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const runScript = async (script: string, args: string[] = []) => {
    try {
      await fetch('http://localhost:8000/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script, args })
      })
      setIsRunning(true)
    } catch (e) {
      console.error(e)
    }
  }

  const handleTrain = () => {
    const args = [
      '--data', trainArgs.data,
      '--out-dir', trainArgs.out_dir,
      '--side', trainArgs.side,
      '--lookback', trainArgs.lookback,
      '--zscore-window', trainArgs.zscore_window
    ]
    if (trainArgs.use_gpu) args.push('--use-gpu')
    if (trainArgs.recency_weight) args.push('--recency-weight')
    
    runScript('train_pipeline/train_ensemble_gpu.py', args)
  }

  const handleBacktest = () => {
    const args = [
      '--data', backtestArgs.data,
      '--model-dir', backtestArgs.model_dir,
      '--side', backtestArgs.side,
      '--lookback', backtestArgs.lookback,
      '--threshold', backtestArgs.threshold
    ]
    
    runScript('train_pipeline/walk_forward_backtest.py', args)
  }

  const stopScript = async () => {
    try {
      await fetch('http://localhost:8000/api/stop', { method: 'POST' })
      setIsRunning(false)
    } catch (e) {
      console.error(e)
    }
  }

  const handleEnvChange = (key: string, value: string) => {
    setConfig(prev => ({
      ...prev,
      env: { ...prev.env, [key]: value }
    }))
  }

  const handleModelChange = (key: string, value: any) => {
    let parsedValue = value;
    if (value === 'true') parsedValue = true;
    if (value === 'false') parsedValue = false;
    if (!isNaN(Number(value)) && value !== '') parsedValue = Number(value);

    setConfig(prev => ({
      ...prev,
      model: { ...prev.model, [key]: parsedValue }
    }))
  }

  const saveConfig = async () => {
    try {
      await fetch('http://localhost:8000/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          env_vars: config.env,
          model_config: config.model
        })
      })
      alert("Configuration saved successfully!")
    } catch (e) {
      console.error(e)
      alert("Failed to save configuration.")
    }
  }

  return (
    <div className="flex h-screen bg-neutral-950 text-neutral-50 overflow-hidden font-sans">
      {/* Sidebar */}
      <div className="w-64 border-r border-neutral-800 bg-neutral-900 flex flex-col">
        <div className="p-4 border-b border-neutral-800">
          <h1 className="text-xl font-bold text-amber-500 tracking-tight">XAUUSD Bot</h1>
          <p className="text-xs text-neutral-400 mt-1">Control Panel</p>
        </div>
        <nav className="flex-1 p-2 space-y-1">
          <NavItem 
            active={activeTab === 'settings'} 
            onClick={() => setActiveTab('settings')} 
            icon={<Settings size={18} />} 
            label="Settings" 
          />
          <NavItem 
            active={activeTab === 'training'} 
            onClick={() => setActiveTab('training')} 
            icon={<Activity size={18} />} 
            label="Training Pipeline" 
          />
          <NavItem 
            active={activeTab === 'live'} 
            onClick={() => setActiveTab('live')} 
            icon={<Play size={18} />} 
            label="Live Trading" 
          />
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <main className="flex-1 overflow-auto p-6">
          <div className="max-w-4xl mx-auto">
            {activeTab === 'settings' && (
              <div className="space-y-6 pb-20">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-semibold">Configuration</h2>
                  <button 
                    onClick={saveConfig}
                    className="flex items-center gap-2 bg-amber-600 hover:bg-amber-500 text-white px-4 py-2 rounded-md font-medium transition-colors"
                  >
                    <Save size={16} />
                    Save Changes
                  </button>
                </div>
                
                <div className="p-4 border border-neutral-800 rounded-lg bg-neutral-900">
                  <h3 className="text-lg font-medium text-neutral-300 mb-4">MT5 Credentials (.env)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(config.env).map(([key, value]) => (
                      <div key={key} className="flex flex-col gap-1">
                        <label className="text-xs font-medium text-neutral-400">{key}</label>
                        <input 
                          type="text" 
                          value={value} 
                          onChange={(e) => handleEnvChange(key, e.target.value)}
                          className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:outline-none focus:border-amber-500"
                        />
                      </div>
                    ))}
                    {Object.keys(config.env).length === 0 && (
                      <p className="text-sm text-neutral-500 italic">No environment variables loaded.</p>
                    )}
                  </div>
                </div>
                
                <div className="p-4 border border-neutral-800 rounded-lg bg-neutral-900">
                  <h3 className="text-lg font-medium text-neutral-300 mb-4">Model Config (model_config.json)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(config.model).map(([key, value]) => (
                      <div key={key} className="flex flex-col gap-1">
                        <label className="text-xs font-medium text-neutral-400">{key}</label>
                        <input 
                          type="text" 
                          value={value?.toString() || ''} 
                          onChange={(e) => handleModelChange(key, e.target.value)}
                          className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
                        />
                      </div>
                    ))}
                    {Object.keys(config.model).length === 0 && (
                      <p className="text-sm text-neutral-500 italic">No model config loaded.</p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'training' && (
              <div className="space-y-6 pb-20">
                <h2 className="text-2xl font-semibold">Training Pipeline</h2>
                
                <div className="p-4 border border-neutral-800 rounded-lg bg-neutral-900">
                  <h3 className="text-lg font-medium text-amber-500 mb-4">Train Ensemble</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Data CSV Path</label>
                      <input type="text" value={trainArgs.data} onChange={e => setTrainArgs({...trainArgs, data: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-amber-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Output Directory</label>
                      <input type="text" value={trainArgs.out_dir} onChange={e => setTrainArgs({...trainArgs, out_dir: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-amber-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Side (long/short)</label>
                      <input type="text" value={trainArgs.side} onChange={e => setTrainArgs({...trainArgs, side: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-amber-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Lookback</label>
                      <input type="text" value={trainArgs.lookback} onChange={e => setTrainArgs({...trainArgs, lookback: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-amber-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Z-Score Window</label>
                      <input type="text" value={trainArgs.zscore_window} onChange={e => setTrainArgs({...trainArgs, zscore_window: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-amber-500 outline-none" />
                    </div>
                    <div className="flex items-center gap-4 mt-4">
                      <label className="flex items-center gap-2 text-sm text-neutral-300">
                        <input type="checkbox" checked={trainArgs.use_gpu} onChange={e => setTrainArgs({...trainArgs, use_gpu: e.target.checked})} className="rounded bg-neutral-900 border-neutral-700 text-amber-500 focus:ring-amber-500" />
                        Use GPU
                      </label>
                      <label className="flex items-center gap-2 text-sm text-neutral-300">
                        <input type="checkbox" checked={trainArgs.recency_weight} onChange={e => setTrainArgs({...trainArgs, recency_weight: e.target.checked})} className="rounded bg-neutral-900 border-neutral-700 text-amber-500 focus:ring-amber-500" />
                        Recency Weighting
                      </label>
                    </div>
                  </div>
                  <button 
                    disabled={isRunning}
                    onClick={handleTrain}
                    className="w-full bg-amber-600 hover:bg-amber-500 text-white font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                  >
                    Run Training
                  </button>
                </div>

                <div className="p-4 border border-neutral-800 rounded-lg bg-neutral-900">
                  <h3 className="text-lg font-medium text-sky-500 mb-4">Walk-Forward Backtest</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Data CSV Path</label>
                      <input type="text" value={backtestArgs.data} onChange={e => setBacktestArgs({...backtestArgs, data: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-sky-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Model Directory</label>
                      <input type="text" value={backtestArgs.model_dir} onChange={e => setBacktestArgs({...backtestArgs, model_dir: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-sky-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Side</label>
                      <input type="text" value={backtestArgs.side} onChange={e => setBacktestArgs({...backtestArgs, side: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-sky-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Lookback</label>
                      <input type="text" value={backtestArgs.lookback} onChange={e => setBacktestArgs({...backtestArgs, lookback: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-sky-500 outline-none" />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-neutral-400">Confidence Threshold</label>
                      <input type="text" value={backtestArgs.threshold} onChange={e => setBacktestArgs({...backtestArgs, threshold: e.target.value})} className="bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm focus:border-sky-500 outline-none" />
                    </div>
                  </div>
                  <button 
                    disabled={isRunning}
                    onClick={handleBacktest}
                    className="w-full bg-sky-600 hover:bg-sky-500 text-white font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                  >
                    Run Backtest
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'live' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold flex items-center justify-between">
                  <span>Live Trading</span>
                  {isRunning && <span className="flex h-3 w-3 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                  </span>}
                </h2>
                
                <div className="p-4 border border-neutral-800 rounded-lg bg-neutral-900 flex flex-col sm:flex-row gap-4 items-center justify-between">
                  <div>
                    <h3 className="text-lg font-medium text-neutral-300">Ensemble Trader</h3>
                    <p className="text-sm text-neutral-500 mt-1">Runs the live trading bot using the ensemble model on MetaTrader 5.</p>
                  </div>
                  <div className="flex gap-2">
                    <button 
                      onClick={() => stopScript()}
                      disabled={!isRunning}
                      className="bg-red-900/50 hover:bg-red-900/80 text-red-200 border border-red-800 font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                    >
                      Stop
                    </button>
                    <button 
                      onClick={() => runScript('live_ensemble_trading.py')}
                      disabled={isRunning}
                      className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                    >
                      Start Live Bot
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Terminal / Logs */}
        <div className="h-64 border-t border-neutral-800 bg-[#0a0a0a] flex flex-col">
          <div className="px-4 py-2 border-b border-neutral-800 flex items-center gap-2 bg-neutral-900">
            <SquareTerminal size={16} className="text-neutral-400" />
            <span className="text-xs font-mono text-neutral-400">Terminal Output</span>
            {isRunning && <span className="ml-auto text-[10px] uppercase tracking-wider text-emerald-500 font-semibold">Running</span>}
          </div>
          <div className="flex-1 overflow-auto p-4 font-mono text-xs text-neutral-300 whitespace-pre-wrap">
            {logs.length === 0 ? (
              <span className="text-neutral-600 italic">No output yet. Run a script to see logs.</span>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="mb-1 leading-relaxed">{log}</div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  )
}

function NavItem({ active, onClick, icon, label }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
        active 
          ? "bg-amber-500/10 text-amber-500" 
          : "text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
      )}
    >
      {icon}
      {label}
    </button>
  )
}