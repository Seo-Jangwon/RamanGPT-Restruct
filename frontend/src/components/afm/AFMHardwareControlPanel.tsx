import { useState } from 'react'
import { Terminal, Send, Loader2, CheckCircle, XCircle, Code, Play } from 'lucide-react'
import axios from 'axios'

interface CommandHistory {
  command: string
  response: string
  success: boolean
  timestamp: Date
  structuredCommand?: any
}

export default function AFMHardwareControlPanel() {
  const [command, setCommand] = useState('')
  const [isExecuting, setIsExecuting] = useState(false)
  const [history, setHistory] = useState<CommandHistory[]>([])

  const exampleCommands = [
    'Move stage to X=100, Y=200',
    'Set laser power to 10 mW',
    'Map this 10×10 μm region with 1 μm steps',
    'Acquire spectrum with 2 second exposure',
    'Move stage 50 microns to the right',
    'Turn laser on',
    'Set grating to 1200 gr/mm'
  ]

  const handleExecute = async () => {
    if (!command.trim()) return

    setIsExecuting(true)
    const startTime = new Date()

    try {
      // Stub: Natural language → structured RAONSpec commands
      // Use existing command schema (from MD call hierarchy)
      const response = await axios.post('/api/hardware-command', {
        command: command
      })

      // Parse structured command (stub - would come from backend)
      const structuredCommand = parseCommandToStructured(command)

      setHistory(prev => [{
        command,
        response: response.data.message || 'Command executed successfully',
        success: true,
        timestamp: startTime,
        structuredCommand
      }, ...prev])

      setCommand('')
    } catch (error: any) {
      setHistory(prev => [{
        command,
        response: error.response?.data?.detail || 'Failed to execute command',
        success: false,
        timestamp: startTime
      }, ...prev])
    } finally {
      setIsExecuting(false)
    }
  }

  // Stub: Parse natural language to structured command
  // This would match the actual pipeline described in MD files
  const parseCommandToStructured = (cmd: string): any => {
    const cmdLower = cmd.toLowerCase()
    
    if (cmdLower.includes('map')) {
      // Extract mapping parameters
      const regionMatch = cmd.match(/(\d+)[×x](\d+)\s*μm/)
      const stepMatch = cmd.match(/(\d+(?:\.\d+)?)\s*μm\s*steps?/)
      
      return {
        type: 'mapping',
        params: {
          region: {
            x: regionMatch ? parseFloat(regionMatch[1]) : 10,
            y: regionMatch ? parseFloat(regionMatch[2]) : 10
          },
          stepSize: stepMatch ? parseFloat(stepMatch[1]) : 0.5,
          mode: '2D'
        },
        pipeline: [
          'MainForm → StageControl → AndorCCD → SpectrumData (dummy)'
        ]
      }
    } else if (cmdLower.includes('move') && cmdLower.includes('stage')) {
      const xMatch = cmd.match(/x\s*=\s*(\d+(?:\.\d+)?)/i)
      const yMatch = cmd.match(/y\s*=\s*(\d+(?:\.\d+)?)/i)
      const zMatch = cmd.match(/z\s*=\s*(\d+(?:\.\d+)?)/i)
      
      return {
        type: 'stage_move',
        params: {
          x: xMatch ? parseFloat(xMatch[1]) : null,
          y: yMatch ? parseFloat(yMatch[1]) : null,
          z: zMatch ? parseFloat(zMatch[1]) : null
        },
        pipeline: [
          'MainForm → StageControl.moveAbsolute() → Hardware'
        ]
      }
    } else if (cmdLower.includes('laser')) {
      const powerMatch = cmd.match(/(\d+(?:\.\d+)?)\s*mw/i)
      return {
        type: 'laser_control',
        params: {
          power: powerMatch ? parseFloat(powerMatch[1]) : null,
          enable: cmdLower.includes('on') || cmdLower.includes('off')
        },
        pipeline: [
          'MainForm → Spectrometer.setLaserPower() → Hardware'
        ]
      }
    } else if (cmdLower.includes('acquire')) {
      const expMatch = cmd.match(/(\d+(?:\.\d+)?)\s*second/i)
      return {
        type: 'acquisition',
        params: {
          exposure: expMatch ? parseFloat(expMatch[1]) : 1.0,
          mode: 'single'
        },
        pipeline: [
          'MainForm → AndorCCDControl.AcquireSingle() → AndorSDK → Hardware'
        ]
      }
    }
    
    return {
      type: 'unknown',
      params: {},
      pipeline: ['Command parsing stub']
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleExecute()
    }
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header - Gemini Style */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-light text-gray-900 tracking-tight">
          LLM-Based Hardware Control
        </h1>
        <p className="text-gray-600 font-light max-w-2xl mx-auto">
          Control hardware with natural language commands.
        </p>
      </div>

      {/* Command Input */}
      <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Natural Language Command
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="e.g., Map this 10×10 μm region with 0.5 μm steps..."
            className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-afm-primary-500 focus:border-afm-primary-500"
          />
          <button
            onClick={handleExecute}
            disabled={isExecuting || !command.trim()}
            className="px-6 py-3 bg-afm-primary-500 text-white rounded-xl hover:bg-afm-primary-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2 font-medium"
          >
            {isExecuting ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>

      {/* Example Commands */}
      <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
        <p className="text-sm font-medium text-gray-700 mb-3">Example Commands:</p>
        <div className="flex flex-wrap gap-2">
          {exampleCommands.map((cmd, idx) => (
            <button
              key={idx}
              onClick={() => setCommand(cmd)}
              className="px-3 py-1.5 bg-afm-primary-50 border border-afm-primary-200 text-afm-primary-700 text-sm rounded-lg hover:bg-afm-primary-100 transition-colors"
            >
              {cmd}
            </button>
          ))}
        </div>
      </div>

      {/* Command History */}
      <div className="bg-white/80 backdrop-blur-sm border border-gray-200/60 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
        <div className="flex items-center gap-2 mb-4">
          <Terminal className="w-5 h-5 text-afm-primary-500" />
          <h3 className="font-semibold text-gray-900">Command History</h3>
        </div>

        {history.length === 0 ? (
          <div className="bg-gray-50 rounded-xl p-12 text-center">
            <Terminal className="w-16 h-16 text-gray-300 mx-auto mb-3" />
            <p className="text-sm text-gray-500">No commands executed yet</p>
            <p className="text-xs text-gray-400 mt-1">Try one of the example commands above</p>
          </div>
        ) : (
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {history.map((item, idx) => (
              <div
                key={idx}
                className={`rounded-xl p-4 border-2 ${
                  item.success
                    ? 'bg-green-50 border-green-200'
                    : 'bg-red-50 border-red-200'
                }`}
              >
                <div className="flex items-start gap-3">
                  {item.success ? (
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 mb-1">
                      $ {item.command}
                    </p>
                    <p className={`text-sm mb-2 ${
                      item.success ? 'text-green-800' : 'text-red-800'
                    }`}>
                      {item.response}
                    </p>
                    
                    {/* Structured Command Preview */}
                    {item.structuredCommand && (
                      <div className="mt-3 p-3 bg-white border border-gray-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <Code className="w-4 h-4 text-gray-600" />
                          <p className="text-xs font-semibold text-gray-700">Structured Command:</p>
                        </div>
                        <pre className="text-xs text-gray-600 bg-gray-50 p-2 rounded overflow-x-auto">
                          {JSON.stringify(item.structuredCommand, null, 2)}
                        </pre>
                        {item.structuredCommand.pipeline && (
                          <div className="mt-2 pt-2 border-t border-gray-200">
                            <p className="text-xs font-semibold text-gray-700 mb-1">Pipeline Call Chain:</p>
                            <div className="flex items-center gap-1 text-xs text-gray-600">
                              {item.structuredCommand.pipeline.map((step: string, i: number) => (
                                <span key={i} className="flex items-center">
                                  {i > 0 && <span className="mx-1">→</span>}
                                  <code className="bg-gray-100 px-1.5 py-0.5 rounded">{step}</code>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    <p className="text-xs text-gray-500 mt-2">
                      {item.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="bg-afm-primary-50 border-2 border-afm-primary-200 rounded-xl p-6">
        <p className="text-sm font-semibold text-afm-primary-900 mb-3">Quick Actions</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setCommand('Turn laser on')}
            className="px-3 py-2 bg-white border border-afm-primary-200 text-afm-primary-900 text-sm rounded-lg hover:bg-afm-primary-100 transition-colors"
          >
            🔦 Laser On
          </button>
          <button
            onClick={() => setCommand('Turn laser off')}
            className="px-3 py-2 bg-white border border-afm-primary-200 text-afm-primary-900 text-sm rounded-lg hover:bg-afm-primary-100 transition-colors"
          >
            🔦 Laser Off
          </button>
          <button
            onClick={() => setCommand('Move to home position')}
            className="px-3 py-2 bg-white border border-afm-primary-200 text-afm-primary-900 text-sm rounded-lg hover:bg-afm-primary-100 transition-colors"
          >
            🏠 Home
          </button>
          <button
            onClick={() => setCommand('Acquire single spectrum')}
            className="px-3 py-2 bg-white border border-afm-primary-200 text-afm-primary-900 text-sm rounded-lg hover:bg-afm-primary-100 transition-colors"
          >
            <Play className="w-4 h-4 inline mr-1" />
            Acquire
          </button>
        </div>
      </div>
    </div>
  )
}

